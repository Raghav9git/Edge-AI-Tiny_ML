import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset

data = tf.keras.datasets.mnist

(training_images, training_labels), (validation_images, validation_labels) = data.load_data()

# Normalize pixel values to the range 0 to 1
# converting 0-255 scale of pixels into 0-1 scale 
training_images = training_images / 255.0
validation_images = validation_images / 255.0


# Display sample MNIST images

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(training_images[i], cmap="gray")
    plt.title(f"Label: {training_labels[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# Define the neural network

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),    
    tf.keras.layers.Dense(20, activation="relu"),   # for hidden layer with 20 neurons
    tf.keras.layers.Dense(10, activation="softmax") # for output layer with 10 neurons
])

# Configure the model for training

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model

model.fit(
    training_images,
    training_labels,
    epochs=20
)

# Evaluate the model on unseen images

validation_loss, validation_accuracy = model.evaluate(
    validation_images,
    validation_labels,
    verbose=2
)

print("\nValidation accuracy:", validation_accuracy)
print("Validation loss:", validation_loss)

# Generate predictions for the validation images

classifications = model.predict(
    validation_images,
    verbose=0
)

print("\nPrediction probabilities:")
print(classifications[0])

# Find the predicted digit

predicted_digit = np.argmax(classifications[0])

print("\nPredicted digit:", predicted_digit)
print("Actual digit:", validation_labels[0])

# Display the model confidence

confidence = classifications[0][predicted_digit]

print("Confidence:", confidence)
print("Confidence percentage:", confidence * 100)

# Display predicted validation images

plt.figure(figsize=(10, 5))

for i in range(10):
    predicted_digit = np.argmax(classifications[i])

    plt.subplot(2, 5, i + 1)
    plt.imshow(validation_images[i], cmap="gray")
    plt.title(
        f"Predicted: {predicted_digit}\nActual: {validation_labels[i]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

# Display the number of weights and biases

print("\nLayer 1 weights:", model.layers[1].get_weights()[0].shape)
print("Layer 1 biases:", model.layers[1].get_weights()[1].shape)

print("Layer 2 weights:", model.layers[2].get_weights()[0].shape)
print("Layer 2 biases:", model.layers[2].get_weights()[1].shape)