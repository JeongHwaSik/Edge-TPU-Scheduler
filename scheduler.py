import argparse
import logging
import os
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image

from pycoral.utils.dataset import read_label_file
from pycoral.adapters import common
from pycoral.adapters import classify

import tflite_runtime.interpreter as tflite
from pipe import Pipe

_EDGETPU_SHARED_LIB = "libedgetpu.so.1"

# models are already quantized and co-compiled: 'model_quant_edgetpu.tflite' file
models = {
    "mobilenet_v2": {
        "path": "mobilenet_v2_1.0_224_quant_edgetpu.tflite",
        "wcet": 75,
    },
    "inception_v2": {
        "path": "inception_v2_224_quant_edgetpu.tflite",
        "wcet": 105,
    },
    "efficientnet_m": {
        "path": "efficientnet-edgetpu-M_quant_edgetpu.tflite",
        "wcet": 180,
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, algorithm):
        self.algorithm = algorithm  # Scheduling algorithm (FIFO, RM, EDF)
        self.pipe_path = Path("/tmp/scheduler_pipe")
        self.pipe = Pipe(str(self.pipe_path))
        self.lock = threading.Lock()
        self.labels = read_label_file("imagenet_labels.txt")
        self.interpreters = self.load_interpreters()
        self.ready_queue = []

    def load_interpreters(self):
        # Load and initialize the interpreters for each model
        delegates = [tflite.load_delegate(_EDGETPU_SHARED_LIB)] # make inference objects for TPU
        
        # MobileNet interpreter
        mobilenet_interpreter = tflite.Interpreter(
            model_path = models["mobilenet_v2"]["path"], experimental_delegates=delegates
        )
        # Inception interpreter
        inception_interpreter = tflite.Interpreter(
            model_path = models["inception_v2"]["path"], experimental_delegates=delegates
        )
        # EfficientNet interpreter
        efficientnet_interpreter = tflite.Interpreter(
            model_path = models["efficientnet_m"]["path"], experimental_delegates=delegates
        )

        # add each models' interpreter in the dictionary
        interpreters = {
            "mobilenet_v2" : mobilenet_interpreter, 
            "inception_v2" : inception_interpreter,
            "efficientnet_m" : efficientnet_interpreter
        }
        return interpreters

    def receive_request_loop(self):
        # Continuously read requests from the pipe
        # Parse each request and add it to the ready queue
        while True:
            request = self.pipe.read()
            parsed_data = self._data_parsing(request)

            self.ready_queue.append(parsed_data)

    def schedule_requests(self):
        # Schedule requests based on the specified algorithm

        # RM (Rate Monotonic)
        if self.algorithm == "RM":
            self.ready_queue.sort(key=lambda list: float(list[3])) # sort by Period Length

        # EDF (Earliest Deadline First)
        elif self.algorithm == "EDF":
            right_now = time.time()
            for idx, data in enumerate(self.ready_queue):
                deadline_left = float(data[4])-right_now # deadline_left = absolute_deadline - right_now
                self.ready_queue[idx].append(deadline_left) 

            self.ready_queue.sort(key=lambda list: list[6]) # sort by Earliest Deadline
            # and then delete
            for data in self.ready_queue:
                del data[6]

        # DEFAULT = "FIFO"
        else: 
            pass

        # pop out the first element from the ready queue and return it
        return self.ready_queue.pop()

    def execute_request(self, request):
        # Execute the scheduled request
        
        pid, model_name, image_path, period, absolute_deadline, request_time = request
        print(f"[Scheduler] Executing request: {{'client_id': {pid}, 'model_name': {model_name}, \
        'image_file': {image_path}, 'period': {period}, 'absolute deadline': {absolute_deadline}, 'timestamp': {request_time}}}")

        # Get the interpreter of the requested model.
        interpreter = self.interpreters[model_name]
        interpreter.allocate_tensors() 
        input_details = interpreter.get_input_details()[0]
        input_shape = input_details["shape"]
        input_size = (input_shape[2], input_shape[1]) # (224, 224)

        # Process the input image.
        input_data = Image.open(image_path).convert("RGB").resize(input_size, Image.LANCZOS) # (224, 224, 3)
        input_data = np.asarray(input_data)
        input_data = np.expand_dims(input_data, axis=0) # (1, 224, 224, 3)
        
        # Perform Edge TPU computation.
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke() # forward pass

        # Find the class with the highest probability and return it to the client.
        output_details = interpreter.get_output_details()[0]
        output_data = interpreter.get_tensor(output_details["index"])
        predicted = np.argmax(output_data)
        score = self._get_score(output_data) 

        # send data to client
        print(f"[Scheduler] Sending Response: {self.labels[predicted]}, {score}")
        client_pipe = Pipe(f"/tmp/pipe_{pid}", create=False)
        client_pipe.write(
            f"{self.labels[predicted]}, {score}"
        )


    def run(self):
        # Run the scheduler

        # Listening Thread: Keep Listening client request
        listening_thread = threading.Thread(target=self.receive_request_loop)
        listening_thread.start()

        # Main Thread: check, schedule, and execute requests
        while True:
            if len(self.ready_queue) == 0:
                # print("wait")
                time.sleep(0.5)
            else:
                self.lock.acquire() # lock ready_queue for data synchronization
                request = self.schedule_requests()
                self.lock.release()

                print(f"[Scheduler] Received request: {request[0]},{request[1]},{request[2]},{request[3]},{request[4]},{request[5]}")
                self.execute_request(request)
        
    
    def _data_parsing(self, request):
        '''
        data_list[0] : PID
        data_list[1] : model name
        data_list[2] : image path
        data_list[3] : period
        data_list[4] : absolute deadline
        data_list[5] : request time
        '''
        data_list = request[0].split(",")
        PID, model_name, image_path, period, absolute_deadline, request_time =\
         data_list[0], data_list[1], data_list[2], data_list[3], data_list[4], data_list[5]

        return [PID, model_name, image_path, period, absolute_deadline, request_time]


    def _get_score(self, output_data):
        predicted = np.argmax(output_data)
        total = np.sum(output_data)
        # return accuracy(%)
        return output_data[0, predicted] / total



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scheduler")
    parser.add_argument(
        "--algorithm", type=str, required=True, help="Scheduling algorithm"
    )

    args = parser.parse_args()
    scheduler = Scheduler(args.algorithm)
    scheduler.run()
