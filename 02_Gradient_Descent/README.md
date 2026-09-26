# Gradient Descent — Linear Regression

## Overview

This folder demonstrates how a machine learning model can learn its parameters using **Gradient Descent**.

The example uses a simple linear regression problem:

```text
y = wx + b
```

The training data follows:

```text
y = 2x - 1
```

Therefore, the correct parameters are:

```text
w = 2
b = -1
```

Instead of allowing TensorFlow/Keras to hide the training process behind `model.fit()`, this implementation manually defines the model, loss function, gradient calculation, and parameter updates.

---

## Learning Objective

The main objective is to understand what happens during model training:

```text
Initial parameters
       ↓
Make predictions
       ↓
Calculate loss
       ↓
Calculate gradients
       ↓
Update parameters
       ↓
Repeat
       ↓
Parameters converge
```

This is the basic optimization process used to train machine learning models.

---

## Project File

```text
gradient_descent_linear_regression.py
```

The program implements linear regression and trains the model using TensorFlow's automatic differentiation system.

---

## 1. Linear Regression

Linear regression models the relationship between an input `x` and an output `y` using:

```text
y = wx + b
```

Where:

* `x` — input value.
* `y` — target/output value.
* `w` — weight. It controls the slope of the line.
* `b` — bias. It controls the vertical offset/intercept.
* `ŷ` — predicted value produced by the model.

For this project:

```text
y = 2x - 1
```

so:

```text
w = 2
b = -1
```

---

## 2. Loss Function

A model needs a numerical measurement of how wrong its predictions are.

This measurement is called the **loss**.

The implementation uses **Mean Squared Error (MSE)**:

```text
MSE = average((prediction - target)²)
```

Squaring the errors prevents positive and negative errors from cancelling each other.

A smaller MSE means the predictions are generally closer to the target values.

---

## 3. Model

The model is represented by a Python class containing two trainable parameters:

```text
w
b
```

The model calculates:

```text
prediction = w × x + b
```

TensorFlow's `tf.Variable` is used because these values need to change during training.

---

## 4. Gradient

A **gradient** describes how the loss changes when a model parameter changes.

For this model we calculate:

```text
dw = ∂Loss / ∂w
db = ∂Loss / ∂b
```

These values tell Gradient Descent which direction the parameters should move to reduce the loss.

---

## 5. GradientTape

`tf.GradientTape()` is TensorFlow's automatic differentiation mechanism.

It records the mathematical operations involved in calculating the loss and then calculates the derivatives of the loss with respect to the model parameters.

In this project it is used to obtain:

```text
dw
db
```

without manually deriving the equations.

---

## 6. Gradient Descent

Gradient Descent updates the model parameters using:

```text
w = w - learning_rate × dw
b = b - learning_rate × db
```

The process is repeated over multiple epochs.

The goal is to find parameter values that minimize the loss.

---

## 7. Learning Rate

The **learning rate** controls how large each parameter update is.

A small learning rate produces smaller updates and can make training slower.

A large learning rate produces larger updates and can cause the optimization process to overshoot or become unstable.

The implementation uses:

```text
learning_rate = 0.1
```

---

## 8. Epoch

An **epoch** represents one complete training iteration over the provided dataset in this implementation.

The model is trained repeatedly so that its parameters gradually move toward values that minimize the loss.

---

## 9. Convergence

**Convergence** means that the model parameters are approaching stable values.

For this problem:

```text
w → 2
b → -1
loss → 0
```

The project also plots the values of `w` and `b` across training epochs so their convergence can be visualized.

---

## 10. Prediction

After training, the model can be used with a new input.

For example:

```text
x = 10
```

The correct mathematical output is:

```text
y = 2(10) - 1
  = 19
```

The trained model should produce a value very close to `19`.

---

## Relationship to Keras `model.fit()`

In the previous neural-network exercise, training was performed using:

```python
model.fit(xs, ys, epochs=500)
```

Keras automatically handled the training process.

This project opens that process and implements the important components explicitly:

```text
prediction
    ↓
loss
    ↓
gradient calculation
    ↓
parameter update
    ↓
repeat
```

Understanding this process is important because more complex neural networks are trained using the same fundamental optimization idea.

---

## Technologies

* Python
* NumPy
* TensorFlow
* Matplotlib

---

## Key Concepts Learned

* Linear regression
* Weight
* Bias
* Prediction
* Loss
* Mean Squared Error
* Gradient
* Gradient Descent
* Learning rate
* Epoch
* Convergence
* Automatic differentiation
* TensorFlow GradientTape
* Parameter optimization

---

## Running the Project

Activate the project virtual environment:

```bash
source .venv/Scripts/activate
```

Run:

```bash
python 02_Gradient_Descent/gradient_descent_linear_regression.py
```

The program prints the training progress and displays plots showing parameter convergence and loss reduction.
