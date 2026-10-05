import tensorflow as tf
import numpy as np

# Define the model

layer_1 = tf.keras.layers.Dense(
    units=2,
    input_shape=(2,)
)

layer_2 = tf.keras.layers.Dense(
    units=1
)

model = tf.keras.Sequential([
    layer_1,
    layer_2
])

# Compile the model

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.01
)

model.compile(
    optimizer=optimizer,
    loss="mean_squared_error"
)

# Generate training data

xs = []
ys = []

for x1 in range(100):
    for x2 in range(100):
        xs.append([x1, x2])
        ys.append([2 * x1 - 2 * x2 + 2])

xs = np.array(xs, dtype=float)
ys = np.array(ys, dtype=float)

print("Input data shape:")
print(xs.shape)

print("\nTarget data shape:")
print(ys.shape)

# Train the model

model.fit(
    xs,
    ys,
    epochs=20,
    verbose=1
)

# Test the model

test_input = np.array([
    [120.0, -10.0]
])

prediction = model.predict(
    test_input,
    verbose=0
)

expected = 2 * 120 - 2 * (-10) + 2

print("\nInput:")
print("x1 =", test_input[0][0])
print("x2 =", test_input[0][1])

print("\nPredicted output:")
print(prediction[0][0])

print("\nExpected output:")
print(expected)

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