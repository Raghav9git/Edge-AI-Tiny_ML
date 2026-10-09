# Neural Networks

## Overview

This folder documents the progression from simple neural networks to **multi-layer networks and Convolutional Neural Networks (CNNs)** using Python, TensorFlow, and Keras.

The implementations cover three areas:

1. **Regression:** Learning mathematical relationships using single-layer and multi-layer neural networks.
2. **Classification:** Recognizing handwritten digits using the MNIST dataset.
3. **Convolutional Neural Networks:** Classifying clothing images using the Fashion-MNIST dataset.

The goal is to understand how neurons, layers, activation functions, loss functions, optimizers, and training work together to solve increasingly complex machine-learning problems.

## Project Structure

```text
03_Neural_Networks/
├── Regression/
│   ├── first_regression_NN.py
│   ├── multilayer_single_input_nn_regression.py
│   └── multilayer_multi_input_nn_regression.py
│
├── Classification/
│   ├── mnist_classification.py
│   ├── Predicted_labels.png
│   └── Sample_labels.png
│
├── CNN/
│   ├── fashin_mnist_cnn.py
│   ├── fashion_mnist_predictions.png
│   ├── fashion_mnist_training_accuracy.png
│   ├── CNN code.pdf
│   └── intro to cnn.pdf
│
└── README.md
```

## Part 1 — Regression

### Objective

The regression implementations demonstrate how neural networks learn relationships between numerical inputs and outputs.

The initial example uses data following:

```text
y = 2x - 1
```

Examples:

```text
x = 0  →  y = -1
x = 1  →  y = 1
x = 2  →  y = 3
x = 3  →  y = 5
```

The network learns to approximate this relationship from examples rather than being explicitly given the equation.

### Implementations

**First Neural Network**

Implements a single-layer neural network that learns a linear relationship.

**Multi-Layer Single-Input Neural Network**

Uses a hidden layer to learn a mapping from one input variable to a continuous output.

**Multi-Layer Multi-Input Neural Network**

Accepts two input variables and learns their relationship with a continuous output.

### Dense Layer

A **Dense layer** is a neural-network layer in which each neuron is connected to the outputs of the previous layer.

Each connection has a trainable weight, and each neuron has a bias.

### Weight

A **weight** controls how strongly an input contributes to a neuron's output. During training, weights are adjusted to reduce the loss.

### Bias

A **bias** is a trainable value added to the weighted input. It allows a neuron to shift its output independently of its inputs.

### Loss Function

The regression implementations use **Mean Squared Error (MSE)** to measure the difference between predicted and actual numerical values.

### Optimizer

An **optimizer** updates the model's trainable parameters during training. The first neural-network implementation uses Stochastic Gradient Descent (SGD).

## Part 2 — MNIST Classification

### Objective

Classification assigns an input to one of several predefined categories.

The MNIST implementation recognizes handwritten digits from `0` to `9`.

### MNIST Dataset

MNIST contains grayscale images of handwritten digits.

Each image has dimensions of:

```text
28 × 28 pixels
```

The dataset contains:

* 60,000 training images
* 10,000 test images

Each image has a corresponding label identifying the digit it represents.

### Pixel and Normalization

A **pixel** is a small element of an image. Each MNIST pixel initially has an intensity value between `0` and `255`.

The images are normalized by dividing the pixel values by `255.0`, converting them to the range `0–1`.

Normalization places the input values on a convenient numerical scale for training.

### Flatten

A `Flatten` layer converts a two-dimensional image into a one-dimensional sequence of values.

```text
28 × 28 pixels
       ↓
   784 values
```

Flattening changes the arrangement of the data, not its pixel values.

### Model Architecture

The MNIST classifier uses the following architecture:

```text
Input Image (28 × 28)
         |
         v
      Flatten
         |
         v
Dense Layer (20 neurons, ReLU)
         |
         v
Dense Layer (10 neurons, Softmax)
         |
         v
Predicted Digit (0–9)
```

### Hidden Layer

A **hidden layer** is a layer between the input and output layers.

The MNIST model uses 20 neurons in its hidden Dense layer to learn intermediate patterns from the input data.

### Activation Function

An **activation function** determines a neuron's output after its weighted input and bias have been calculated.

Activation functions introduce non-linearity, allowing neural networks to learn more complex relationships.

**ReLU** is defined as:

```text
ReLU(x) = max(0, x)
```

Therefore:

```text
ReLU(-3) = 0
ReLU(2)  = 2
```

The MNIST model uses ReLU in its hidden layer.

### Softmax

The output layer contains 10 neurons, one for each digit class.

The **Softmax** activation function converts the output values into probabilities that sum to approximately `1.0`.

The class with the highest probability is selected as the predicted digit.

### Loss Function and Optimizer

The classifier uses:

* **Sparse categorical cross-entropy:** Measures classification error when labels are integer class IDs.
* **Adam:** Updates the model parameters using gradient information during training.

### Prediction and Accuracy

The model generates a probability for each digit class. NumPy's `argmax()` identifies the index of the highest probability, giving the predicted digit.

**Accuracy** measures the proportion of correctly classified images.

The implementation also displays sample images and compares predicted labels with actual labels.

## Part 3 — Convolutional Neural Networks (CNNs)

### Objective

This implementation introduces **Convolutional Neural Networks (CNNs)** for image classification using the Fashion-MNIST dataset.

Unlike a basic Dense network, a CNN uses convolutional filters to learn spatial features from images, such as edges, shapes, and textures.

### Fashion-MNIST Dataset

Fashion-MNIST contains 70,000 grayscale images of clothing items across 10 categories.

Each image has dimensions of `28 × 28` pixels.

The dataset contains:

* 60,000 training images
* 10,000 test images

The categories include T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots.

### Image Preprocessing

The images undergo two preprocessing operations:

**Normalization:** Pixel values are converted from `0–255` to `0–1`.

**Channel dimension:** A final dimension is added to represent the single grayscale channel.

The resulting input shape is:

```text
(number of images, 28, 28, 1)
```

### CNN Architecture

The Fashion-MNIST classifier uses two convolutional layers, two max-pooling layers, and two Dense layers.

```text
Input Image (28 × 28 × 1)
          |
          v
Conv2D (64 filters, 3 × 3, ReLU)
          |
          v
MaxPooling2D (2 × 2)
          |
          v
Conv2D (64 filters, 3 × 3, ReLU)
          |
          v
MaxPooling2D (2 × 2)
          |
          v
        Flatten
          |
          v
Dense (128 neurons, ReLU)
          |
          v
Dense (10 neurons, Softmax)
          |
          v
Predicted Clothing Category
```

### Convolution

A **convolutional layer** applies learnable filters across an image or feature map.

Each filter processes small regions of the input and learns to respond to useful visual patterns.

`Conv2D(64, (3, 3))` specifies 64 filters, each with a spatial size of `3 × 3`.

### Feature Maps

A **feature map** is produced when a convolutional filter processes an input.

Different filters can learn to detect different visual patterns. Deeper layers can combine simpler features into more complex representations.

### Max Pooling

`MaxPooling2D((2, 2))` divides a feature map into small regions and retains the maximum value from each region.

This reduces the spatial dimensions of the feature maps, decreasing the amount of data passed to subsequent layers.

### Flatten and Dense Layers

The `Flatten` layer converts the final feature maps into a one-dimensional vector.

The Dense layer with 128 neurons learns combinations of the extracted features. The final Dense layer contains 10 neurons, corresponding to the 10 clothing categories.

### Training Configuration

The CNN uses:

* **Adam:** Optimizer for updating trainable parameters.
* **Sparse categorical cross-entropy:** Loss function for integer-labelled, multi-class classification.
* **Softmax:** Output activation that produces class probabilities.
* **Accuracy:** Metric for measuring correct predictions.
* **Early stopping:** Callback used to stop training when validation loss stops improving.

### Training and Evaluation

The initial 10-epoch experiment achieved approximately **90.07% test accuracy**.

A subsequent 50-epoch experiment produced approximately:

| Metric            | Result |
| ----------------- | -----: |
| Training accuracy |  99.3% |
| Test accuracy     |  90.5% |

These results demonstrate a gap between training and test performance.

### Overfitting

**Overfitting** occurs when a model learns its training examples very well but fails to achieve comparable performance on unseen data.

In the 50-epoch experiment, training accuracy continued to rise while test accuracy remained around 90–91%.

This indicated that additional training was not providing a comparable improvement in generalization.

### Early Stopping

Early stopping monitors validation performance and stops training when the monitored metric no longer improves.

The implementation uses:

```python
tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

* `monitor="val_loss"` monitors the loss on the validation data.
* `patience=5` allows five consecutive epochs without improvement before stopping.
* `restore_best_weights=True` restores the weights from the epoch with the lowest monitored loss.

The maximum epoch count can remain at 50; early stopping determines whether training should finish sooner.

**Evaluation note:** The current implementation uses the 10,000-image test split returned by Fashion-MNIST as the data monitored during training. For a rigorous final evaluation, a separate validation set should be created from the training data, with the official test set reserved for the final evaluation.

### Saved Outputs

The CNN implementation generates two visualizations:

**`fashion_mnist_predictions.png`**

Displays sample clothing images alongside their predicted and actual categories.

**`fashion_mnist_training_accuracy.png`**

Plots training accuracy and held-out dataset accuracy across epochs, making it easier to observe learning progress and identify a possible generalization gap.

## Regression vs Classification

| Feature           | Regression                | Classification                   |
| ----------------- | ------------------------- | -------------------------------- |
| Goal              | Predict a numerical value | Predict a category               |
| Example           | `y = 2x - 1`              | Digit or clothing category       |
| Output            | Continuous value          | Class probabilities              |
| Example loss      | Mean Squared Error        | Sparse categorical cross-entropy |
| Output activation | Depends on the task       | Softmax for these classifiers    |

## Relationship to Previous Projects

The implementations build on the concepts introduced in the earlier machine-learning and gradient-descent projects.

```text
Linear Regression
       |
       v
Loss Functions
       |
       v
Gradient Descent
       |
       v
Single-Layer Neural Network
       |
       v
Multi-Layer Neural Networks
       |
       v
Activation Functions
       |
       v
Classification
       |
       v
CNNs and Image Classification
       |
       v
Model Evaluation and Overfitting
```

The underlying training process remains similar:

```text
Input
  |
  v
Prediction
  |
  v
Loss Calculation
  |
  v
Gradient Calculation
  |
  v
Parameter Updates
  |
  v
Repeat
```

CNNs extend these fundamentals by learning spatial features from image data.

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib

## Running the Implementations

Activate the project's virtual environment from the repository root:

```bash
source .venv/Scripts/activate
```

Run the regression implementation:

```bash
python 03_Neural_Networks/Regression/first_regression_NN.py
```

Run the MNIST classifier:

```bash
python 03_Neural_Networks/Classification/mnist_classification.py
```

Run the Fashion-MNIST CNN:

```bash
python 03_Neural_Networks/CNN/fashin_mnist_cnn.py
```

The CNN program trains the model, evaluates its performance, displays sample predictions, and saves the generated visualizations in the current working directory.

## Key Concepts Learned

* Neural networks and neurons
* Dense layers
* Weights and biases
* Input, hidden, and output layers
* Activation functions
* ReLU and Softmax
* Regression and classification
* MNIST and Fashion-MNIST
* Image pixels and normalization
* Flattening
* Convolutional filters and feature maps
* Max pooling
* Loss functions
* Mean Squared Error
* Sparse categorical cross-entropy
* Optimizers and Adam
* Epochs and training
* Prediction probabilities and `argmax`
* Accuracy and model evaluation
* Overfitting and early stopping

## Next Steps

The next stages of this learning path will focus on model optimization, quantization, TensorFlow Lite, and embedded inference.

These topics build toward the broader goal of deploying machine-learning models on resource-constrained hardware for **Edge AI and TinyML applications**.
