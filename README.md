# Edge AI & TinyML

A hands-on learning and implementation repository focused on **Edge AI, TinyML, neural networks, model optimization, and machine learning on resource-constrained embedded devices**.

This repository documents my progression from fundamental machine learning concepts to deploying optimized ML models on edge devices. The focus is on understanding the underlying concepts and implementing them through practical experiments and projects.

## Objective

The goal of this repository is to build a strong foundation in:

* Machine Learning fundamentals
* Neural Networks and Deep Learning
* Computer Vision
* Sensor-based Machine Learning
* Feature Extraction
* Model Optimization and Quantization
* TensorFlow Lite
* TinyML
* Embedded Machine Learning
* On-device inference

The long-term objective is to bridge **Machine Learning and Embedded Systems**, enabling ML models to run locally on microcontrollers and other resource-constrained edge devices.

## Repository Structure

```text
Edge-AI-TinyML/
│
├── 01_ML_Fundamentals/
│
├── 02_Gradient_Descent/
│
├── 03_Neural_Networks/
│
├── 04_Computer_Vision/
│
├── 05_Sensor_Data/
│
├── 06_Feature_Extraction/
│
├── 07_Model_Optimization/
│
├── 08_TensorFlow_Lite/
│
├── 09_TinyML/
│
├── 10_Embedded_Inference/
│
└── 11_Final_Project/
```

The repository is being developed progressively. Each section contains implementations and experiments completed as part of the learning process.

## Current Progress

### Machine Learning Fundamentals

Covered concepts include:

* Machine learning paradigm
* Loss functions
* Mean Squared Error
* Linear regression
* Parameter estimation
* Gradient descent
* Learning rate
* Model convergence

Implemented:

* Manual gradient descent for linear regression
* TensorFlow-based training using `GradientTape`
* Parameter convergence visualization

### Neural Networks

Implemented neural networks using TensorFlow and Keras.

Topics covered:

* Neurons
* Dense layers
* Weights and biases
* Activation functions
* Forward propagation
* Loss functions
* Model training
* Multi-layer neural networks
* Single-input regression
* Multi-input regression
* Classification

Implemented examples include:

* Single-layer neural network for linear regression
* Multi-layer neural network for single-input regression
* Multi-layer neural network for multi-input regression
* MNIST handwritten digit classification

### Computer Vision

The next stage of the repository focuses on applying neural networks to image-based problems, including:

* Image representation
* Convolutional Neural Networks
* Convolution layers
* Feature extraction from images
* CNN-based classification

## TinyML Learning Path

The planned progression of this repository follows the complete Edge AI workflow:

```text
Machine Learning
       ↓
Neural Networks
       ↓
Computer Vision / Sensor Data
       ↓
Feature Extraction
       ↓
Model Optimization
       ↓
Quantization
       ↓
TensorFlow Lite
       ↓
TinyML
       ↓
TFLite Micro
       ↓
Embedded C/C++
       ↓
Microcontroller
       ↓
On-device Inference
```

The objective is to understand not only how to train an ML model, but also how to make the model suitable for deployment on devices with limited:

* RAM
* Flash memory
* Processing power
* Energy
* Storage

## Tools and Technologies

### Machine Learning

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow
* Keras
* Matplotlib

### Edge AI / TinyML

* TensorFlow Lite
* TensorFlow Lite Micro
* Model quantization
* Model optimization
* On-device inference

### Embedded Systems

The later stages of the project will integrate:

* Embedded C/C++
* STM32
* ESP32
* Microcontrollers
* Sensors
* UART
* I2C
* SPI

## Planned Final Application

The long-term objective is to develop a **sensor-based TinyML application** capable of performing machine learning inference locally on an embedded device.

A planned workflow is:

```text
IMU / Sensor
     ↓
Data Collection
     ↓
Preprocessing
     ↓
Feature Extraction
     ↓
TinyML Model
     ↓
On-device Inference
     ↓
Classification
     ↓
Embedded Output
```

Possible applications include:

* Human activity recognition
* Motion classification
* Fall detection
* Sensor anomaly detection
* Embedded condition monitoring

The final application will be developed as the required concepts and deployment techniques are covered.

## Engineering Approach

This repository is intentionally implementation-oriented.

For each major concept, the workflow is:

1. Understand the underlying concept.
2. Implement it using Python/TensorFlow.
3. Train and evaluate the model.
4. Inspect model parameters and behavior.
5. Study optimization and deployment constraints.
6. Convert the model for edge deployment.
7. Integrate the model with embedded hardware.

This approach is intended to build an understanding of the complete pipeline rather than treating ML models as black boxes.

## Current Status

This repository is actively under development.

Completed work currently includes:

* Linear regression fundamentals
* Gradient descent
* Parameter optimization
* First neural network
* Multi-layer neural networks
* Multi-input neural networks
* MNIST digit classification

Upcoming work includes:

* Convolutional Neural Networks
* Sensor-based ML
* Feature extraction
* Model optimization
* Quantization
* TensorFlow Lite
* TinyML deployment
* TFLite Micro
* Embedded inference
* Final sensor-based TinyML application

## Why This Repository?

The purpose of this project is to develop practical experience at the intersection of:

**Machine Learning + Embedded Systems + Edge Computing**

Rather than relying exclusively on cloud-based inference, Edge AI enables models to process data locally on the device, which can provide advantages in latency, connectivity requirements, privacy, and resource-constrained applications.

This repository documents my progression toward developing such systems from the model-training stage to embedded deployment.

## Author

**Raghav Sathe**

Electronics and Telecommunication Engineering
Datta Meghe College of Engineering, Navi Mumbai

### Areas of Interest

* Embedded Systems
* Edge AI
* TinyML
* Robotics
* PCB Design
* Sensor Fusion
* Machine Learning
* Computer Vision

## Status

**Active Development**
