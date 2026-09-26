# Neural Networks

## Overview

This folder introduces the transition from simple machine learning models to **neural networks**.

The projects demonstrate two stages:

1. A simple neural network learning a mathematical relationship.
2. A multi-layer neural network performing image classification using the MNIST handwritten-digit dataset.

The goal is to understand how neurons, layers, activation functions, loss functions, and training work together to solve increasingly complex problems.

---

# Project Structure

```text
03_Neural_Networks/
├── first_neural_network.py
├── mnist_classification.py
└── README.md
```

---

# Part 1 — First Neural Network

## Objective

The first project teaches the basic structure of a neural network using a very small dataset.

The data follows:

```text
y = 2x - 1
```

Example:

```text
x = 0  →  y = -1
x = 1  →  y = 1
x = 2  →  y = 3
x = 3  →  y = 5
```

The network learns this relationship from examples rather than being explicitly told the equation.

---

## Model

The network contains one Dense neuron:

```python
tf.keras.layers.Dense(
    units=1,
    input_shape=[1]
)
```

The neuron learns a weight and bias that allow it to approximate:

```text
y = wx + b
```

---

## Dense Layer

A **Dense layer** is a neural-network layer in which neurons are connected to the outputs of the previous layer.

Each connection has a weight.

A neuron combines its inputs using these weights and a bias before applying its activation function.

---

## Weight

A **weight** controls how strongly an input contributes to a neuron's output.

During training, the weights are adjusted to reduce the loss.

---

## Bias

A **bias** is an additional trainable value added to the weighted input.

It allows the neuron to shift its output independently of the input.

---

## Optimizer

An **optimizer** is the algorithm responsible for updating trainable model parameters during training.

The first neural-network project uses:

```text
SGD
```

which means **Stochastic Gradient Descent**.

The optimizer uses gradient information to modify the model parameters so that the loss decreases.

---

## Loss

The first project uses:

```text
Mean Squared Error
```

because the task is regression.

The loss measures how far the model's predictions are from the target values.

---

# Part 2 — MNIST Classification

## Objective

The second project introduces **classification**.

Instead of predicting a continuous numerical value, the model must determine which category an input belongs to.

The example is handwritten-digit recognition.

The possible classes are:

```text
0 1 2 3 4 5 6 7 8 9
```

Therefore, the model has 10 possible output classes.

---

# MNIST Dataset

**MNIST** is a dataset of handwritten digits.

Each image has:

```text
28 × 28 pixels
```

The dataset contains:

```text
60,000 training examples
10,000 test/validation examples
```

Each image has a corresponding label identifying the digit represented in the image.

---

# Pixel

A **pixel** is one small element of an image.

MNIST images contain:

```text
28 × 28 = 784 pixels
```

Each pixel initially contains a value between:

```text
0 and 255
```

representing its grayscale intensity.

---

# Normalization

The images are divided by:

```text
255.0
```

This converts the pixel values from:

```text
0–255
```

to approximately:

```text
0–1
```

This process is called **normalization**.

Normalization puts numerical inputs into a more convenient scale for machine-learning algorithms.

---

# Flatten

An MNIST image starts as a two-dimensional array:

```text
28 × 28
```

A Dense layer receives a one-dimensional sequence of values.

The `Flatten` layer converts:

```text
28 × 28
```

into:

```text
784
```

values.

It does not change the pixel values. It only changes their arrangement.

---

# Neural Network Architecture

The MNIST model uses:

```text
Input
  ↓
Flatten
  ↓
Dense(20, ReLU)
  ↓
Dense(10, Softmax)
  ↓
Output
```

---

# Hidden Layer

A **hidden layer** is a neural-network layer between the input and output.

The MNIST model contains one hidden Dense layer:

```text
20 neurons
```

The hidden layer learns intermediate patterns from the input data.

---

# Neuron

A **neuron** receives inputs, combines them using learned weights and a bias, and produces an output.

Conceptually:

```text
inputs
  ↓
weighted combination
  ↓
activation function
  ↓
output
```

A neural network combines many neurons to learn increasingly useful representations of data.

---

# Activation Function

An **activation function** determines the output produced by a neuron after its weighted input and bias have been calculated.

Activation functions are important because they introduce **non-linearity** into neural networks.

Without useful non-linear transformations, stacking multiple linear operations would still behave like a linear transformation.

---

# ReLU

The hidden layer uses **ReLU**:

```text
ReLU(x) = max(0, x)
```

Therefore:

```text
negative input → 0
positive input → input
```

For example:

```text
ReLU(-3) = 0
ReLU(2)  = 2
```

The MNIST model uses:

```python
activation=tf.nn.relu
```

for its 20-neuron hidden layer.

---

# Output Layer

The final layer contains:

```text
10 neurons
```

because there are 10 possible digit classes.

The neurons correspond to:

```text
0 → digit 0
1 → digit 1
2 → digit 2
...
9 → digit 9
```

---

# Softmax

The output layer uses the **Softmax** activation function.

Softmax converts the model's output values into a probability distribution.

For example:

```text
0 → 0.001
1 → 0.002
2 → 0.000
3 → 0.004
4 → 0.001
5 → 0.003
6 → 0.000
7 → 0.989
8 → 0.000
9 → 0.000
```

The probabilities add up to approximately:

```text
1.0
```

The class with the highest probability is selected as the predicted class.

---

# Classification

**Classification** means assigning an input to one or more predefined categories.

For MNIST:

```text
Input:
handwritten image

Output:
one of 10 digit classes
```

This is a **multi-class classification** problem because there are more than two possible classes.

---

# Probability

A probability represents the model's estimated likelihood for a particular class.

For example:

```text
digit 7 → 0.9989
```

means the model assigns approximately:

```text
99.89%
```

probability to class `7`.

This is the model's prediction confidence, not a guarantee that the prediction is correct.

---

# Argmax

The model produces ten probabilities.

`np.argmax()` finds the position of the largest value.

For example:

```text
[0.01, 0.02, 0.90, 0.07]
```

The largest value is:

```text
0.90
```

at index:

```text
2
```

Therefore:

```python
np.argmax(...)
```

returns:

```text
2
```

For MNIST, the index directly corresponds to the predicted digit.

---

# Loss Function for Classification

Regression and classification use different loss functions.

The MNIST model uses:

```text
sparse_categorical_crossentropy
```

This loss function is suitable when there are multiple classes and the labels are represented as integer class IDs.

For example:

```text
7
```

rather than a one-hot vector such as:

```text
[0,0,0,0,0,0,0,1,0,0]
```

The loss measures how different the model's predicted probability distribution is from the correct class.

---

# Optimizer — Adam

The MNIST model uses:

```text
Adam
```

Adam is an optimization algorithm used to update neural-network parameters during training.

It is designed to adapt the parameter update process using information from previous gradients.

The important concept at this stage is:

```text
prediction
    ↓
loss
    ↓
gradients
    ↓
optimizer
    ↓
updated parameters
```

---

# Epoch

An **epoch** represents one complete pass through the training dataset.

The MNIST model is trained using:

```python
epochs=20
```

This means the training process goes through the training dataset repeatedly for 20 epochs.

---

# Training

Training is the process through which the neural network learns its parameters from labeled examples.

The general process is:

```text
Input image
    ↓
Neural network
    ↓
Prediction
    ↓
Loss calculation
    ↓
Gradient calculation
    ↓
Optimizer updates parameters
    ↓
Repeat
```

This connects directly to the Gradient Descent project in the previous folder.

---

# Prediction

After training, the model can process previously unseen images.

The code:

```python
classifications = model.predict(test_images)
```

generates probability distributions for the test images.

For one image:

```python
classifications[0]
```

contains ten probability values corresponding to digits `0–9`.

---

# Accuracy

**Accuracy** measures the proportion of predictions that are correct.

For example:

```text
95% accuracy
```

means approximately 95 out of every 100 evaluated examples were classified correctly.

Accuracy is used as a metric to evaluate the classifier.

---

# Regression vs Classification

| Feature           | Regression                | Classification                   |
| ----------------- | ------------------------- | -------------------------------- |
| Goal              | Predict a numerical value | Predict a class                  |
| Example           | `y = 2x - 1`              | Digit `0–9`                      |
| Output            | Continuous value          | Class probabilities              |
| Example loss      | MSE                       | Sparse categorical cross-entropy |
| Output activation | Depends on task           | Softmax for this MNIST model     |
| Project           | `first_neural_network.py` | `mnist_classification.py`        |

---

# Relationship to Previous Projects

The learning progression is:

```text
Linear Regression
      ↓
Loss
      ↓
Gradient Descent
      ↓
Single Neural Network
      ↓
Multiple Neurons
      ↓
Hidden Layers
      ↓
Activation Functions
      ↓
Classification
      ↓
MNIST Digit Recognition
```

The underlying training idea remains the same:

```text
make prediction
      ↓
measure loss
      ↓
calculate gradients
      ↓
update parameters
      ↓
repeat
```

The difference is that the model and task have become more complex.

---

# Technologies

* Python
* NumPy
* TensorFlow
* Keras

---

# Running the Projects

Activate the virtual environment:

```bash
source .venv/Scripts/activate
```

Run the first neural-network example:

```bash
python 03_Neural_Networks/first_neural_network.py
```

Run the MNIST classifier:

```bash
python 03_Neural_Networks/mnist_classification.py
```

---

# Key Concepts Learned

* Neural network
* Neuron
* Dense layer
* Weight
* Bias
* Input layer
* Hidden layer
* Output layer
* Activation function
* ReLU
* Softmax
* Flattening
* MNIST
* Pixel
* Normalization
* Classification
* Multi-class classification
* Probability
* Argmax
* Loss function
* Sparse categorical cross-entropy
* Optimizer
* Adam
* Epoch
* Accuracy
* Prediction
* Training
