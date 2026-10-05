import tensorflow as tf
import numpy as np

# Training data for y = 2x - 1

xs = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
ys = np.array([-3, -1, 1, 3, 5, 7], dtype=float)

# Define the model

layer_1 = tf.keras.layers.Dense(units=2, input_shape=(1,))
layer_2 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([layer_1, layer_2])

# Compile the model

model.compile(
    optimizer="sgd",
    loss="mean_squared_error"
)

# Train the model

model.fit(xs, ys, epochs=500, verbose=0)

# Test the model

test_x = np.array([10.0])

prediction = model.predict(test_x, verbose=0)

print("Prediction for x = 10:")
print(prediction[0][0])

print("\nExpected value:")
print(19.0)

# Display learned weights and biases

print("\nLearned parameters:")

print("\nLayer 1 weights:")
print(layer_1.get_weights()[0])

print("\nLayer 1 biases:")
print(layer_1.get_weights()[1])

print("\nLayer 2 weights:")
print(layer_2.get_weights()[0])

print("\nLayer 2 bias:")
print(layer_2.get_weights()[1])