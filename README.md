# Edge TPU Scheduler
</br>

## 1. Motivation
With the rapid improvement in the performance of deep learning, advancements in AI processors have been made, allowing various DNN tasks to be run within these processors. For example, in the NPU of a self-driving car, various computer vision tasks such as object detection, traffic signal detection, and lane tracking can be executed. However, in safety-critical systems such as self-driving cars and smart factories, these tasks must be executed with timing guarantees and must be completed within deadlines; otherwise, it could lead to major accidents. Therefore, flexible scheduling of tasks at the edge device level is essential.

However, the Edge TPU does not have a dedicated scheduler, and when multiple applications try to perform deep learning inference, the Edge TPU operates in a non-preemptive FIFO manner, processing applications in the order in which they request execution. This simple scheduling method cannot guarantee meeting timing constraints when the deep learning inferences of applications have different timing constraints (deadlines). Therefore, I intend to implement various types of schedulers on the Edge TPU.
</br>
</br>

## 2. Preparation

### 2-0. Setup RaspberryPi4


• Setup TPU library (PyCoral) in RaspberryPi4
</br>
</br>
```sudo apt-get install python3-pycoral```
</br>
</br>

• Setup TPU driver (libedgetpu1) in RaspberryPi4
</br>
</br>
```sudo apt-get install libedgetpu1-std```
</br>
</br>

• Connect EdgeTPU with RaspberryPi4
</br>
</br>
<img width="588" alt="Screenshot 2024-06-30 at 2 28 23 PM" src="https://github.com/JeongHwaSik/Edge-TPU-Scheduler/assets/99574746/915b3444-5245-4c01-8372-f26ff65ae128">

### 2-1. Model Training and Compression (e.g. LeNet model for MNIST dataset)

First, implement the LeNet model using TensorFlow and train the model using the MNIST dataset. Since the edge device has limited resources, the resource-intensive training is carried out on a local PC. The trained model is saved in .pb format, and using TensorFlow's TFLiteConverter, the trained model is converted to a .tflite file to ensure that it runs well on the edge device. Additionally, to operate the model on the Edge TPU, quantization needs to be performed. The model is quantized to INT-8, generating the mnist_lenet_quant.tflite file. See training/training_lenet.ipynb.


### 2-2. Edge TPU Co-Compilation

• Follow https://coral.ai/docs/edgetpu/compiler/
</br>
</br>
• Online Compiler for Edge TPU: https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb
</br>
</br>
</br>
<img width="1379" alt="EdgeTPU compile화면" src="https://github.com/JeongHwaSik/Edge-TPU-Scheduler/assets/99574746/55666176-9743-4e6a-ae50-ac2fff868c52">

</br>
Use the Edge TPU online compiler to compile the quantized model. Final files should be 'model_quant_edgetpu.tflite' format. Apply the same process to the EfficientNet, MobileNet, and Inception models. Once all 'model_quant_edgetpu.tflite' files are placed on the Raspberry Pi, the models are ready to run. MobileNetV2, InceptionV2, EfficientNet-M are all trained on ILSVRC2012(ImageNet).
</br>
</br>

## 3. Edge TPU Scheduler Overview

When multiple clients send requests for deep learning tasks to be executed by the scheduler, the scheduler processes these execution requests using one of several algorithms. IPC (Inter-Process Communication) uses named PIPEs. The scheduler operates in one of three ways: RM (Rate Monotonic), EDF (Earliest Deadline First), or FIFO (First In First Out), all assumed to function non-preemptively. The client logs the execution results received from the scheduler into a log file for later use in evaluating the performance of each application.
</br>
</br>

 ## 4. Results

 • Scheduler.py execution with EDF algorithm
 </br>
 </br>
 <img width="770" alt="Screenshot 2024-06-30 at 8 53 43 PM" src="https://github.com/JeongHwaSik/Edge-TPU-Scheduler/assets/99574746/61105e13-f38a-4daa-b705-0df7facfc7be">
</br>
</br>
</br>
 • First client.py execution with MobileNetV2, period=1000, deadline=500
 </br>
 </br>
 <img width="1422" alt="Screenshot 2024-06-30 at 8 53 57 PM" src="https://github.com/JeongHwaSik/Edge-TPU-Scheduler/assets/99574746/158b80cc-224a-444c-9581-9e8a6e469497">
 </br>
 </br>
 </br>
 • Second client.py execution with InceptionV2, period=1000, deadline=500
 </br>
 </br>
 <img width="1426" alt="Screenshot 2024-06-30 at 8 54 14 PM" src="https://github.com/JeongHwaSik/Edge-TPU-Scheduler/assets/99574746/7475c053-d473-4e3d-8a5b-21db1b27bf5a">
 </br>
 </br>
  </br>
 • Third client.py execution with EfficientNet-M, period=1000, deadline=500
 </br>
 </br>
 <img width="1424" alt="Screenshot 2024-06-30 at 8 54 44 PM" src="https://github.com/JeongHwaSik/Edge-TPU-Scheduler/assets/99574746/377f1ed8-ef80-4105-8ed3-b242137b3c68">
 </br>
 </br>
 </br>
 • Evaluation
 </br>
 </br>
 <img width="770" alt="Screenshot 2024-06-30 at 8 57 43 PM" src="https://github.com/JeongHwaSik/Edge-TPU-Scheduler/assets/99574746/c4d35c01-7e51-4638-98dc-a2624901f660">
