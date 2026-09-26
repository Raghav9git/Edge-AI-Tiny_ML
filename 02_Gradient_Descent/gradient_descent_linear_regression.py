import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# TRAINING DATA

xs = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
ys = np.array([-3, -1, 1, 3, 5, 7], dtype=float)

# The true relationship is: y = 2x - 1 so,
# true weight (w) = 2
# true bias   (b) = -1


# DEFINE THE LINEAR REGRESSION MODEL

INITIAL_W = 10.0         # Initial guesses for w 
INITIAL_B = 10.0         # Initial guesses for b

class Model:
    def __init__(self):
        # Create trainable parameters
        self.w = tf.Variable(INITIAL_W)
        self.b = tf.Variable(INITIAL_B)
    def __call__(self, x):
        # Linear regression equation:
        # y = wx + b
        return self.w * x + self.b


# DEFINE THE LOSS FUNCTION

def loss(predicted_y, target_y):
    # Mean Squared Error
    return tf.reduce_mean(
        tf.square(predicted_y - target_y)
    )


# DEFINE THE TRAINING PROCEDURE

def train(model, inputs, outputs, learning_rate):

    # GradientTape tool used to automatically calculate mathematical derivatives
    # so TensorFlow can calculate derivatives later.
    with tf.GradientTape() as tape:

        # Make predictions
        predictions = model(inputs)

        # Calculate current loss
        current_loss = loss(predictions, outputs)

    # Calculate gradients of loss with respect to w and b
    dw, db = tape.gradient(
        current_loss,
        [model.w, model.b]
    )

    # Gradient Descent update:
    #
    # w = w - learning_rate * dw
    # b = b - learning_rate * db

    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)

    return current_loss


# 5. TRAIN THE MODEL

LEARNING_RATE = 0.01

# Create model
model = Model()

# Number of training iterations
epochs = 1000

# Lists to store values during training
list_w = []
list_b = []
losses = []


# Training loop
for epoch in range(epochs):

    # Store current parameters before updating them
    list_w.append(model.w.numpy())
    list_b.append(model.b.numpy())

    # Perform one gradient descent step
    current_loss = train(
        model,
        xs,
        ys,
        LEARNING_RATE
    )

    # Store loss
    losses.append(current_loss.numpy())

    # Print training progress
    print(
        f"Epoch {epoch:02d}: "
        f"w={model.w.numpy():.2f}, "
        f"b={model.b.numpy():.2f}, "
        f"loss={current_loss.numpy():.5f}"
    )


# 6. FINAL MODEL PARAMETERS

print("\nTraining complete!")

print("Final weight (w):", model.w.numpy())
print("Final bias (b):", model.b.numpy())

print("\nExpected values:")
print("Weight (w): 2.0")
print("Bias (b): -1.0")


# 7. MAKE A PREDICTION =

x_test = 10.0

prediction = model(x_test)

print("\nPrediction:")
print(f"For x = {x_test}, predicted y = {prediction.numpy():.4f}")

print("Expected y = 19.0")


# 8. PLOT W AND B DURING TRAINING

TRUE_W = 2.0
TRUE_B = -1.0

x_axis = range(epochs)

plt.plot(x_axis, list_w, label="w")
plt.plot(x_axis, list_b, label="b")

# True values
plt.plot(
    x_axis,
    [TRUE_W] * epochs,
    "--",
    label="True w"
)

plt.plot(
    x_axis,
    [TRUE_B] * epochs,
    "--",
    label="True b"
)

plt.xlabel("Epoch")
plt.ylabel("Parameter Value")
plt.title("Gradient Descent Parameter Convergence")

plt.legend()
plt.show()


# 9. PLOT LOSS DURING TRAINING

plt.plot(x_axis, losses)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss During Training")

plt.show()