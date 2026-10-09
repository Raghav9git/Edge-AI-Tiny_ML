import tensorflow as tf          # provides tools to build and train CNN
import numpy as np               # to process predictions and identify the most likely clothing category.
import matplotlib.pyplot as plt  # Matplotlib to display images and visualize the model's predictions.

# Load the fashin mnist dataset
# Fashion-MNIST is a dataset of grayscale clothing images.

fashin_mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (validation_images, validation_labels) = fashin_mnist.load_data()   # loads the images and their corresponding labels
# training_images       - Images used to train the model                    60,000
# training_labels       - Correct categories for training images            60,000
# validation_images     - Images held out from training for evaluation      10,000
# validation_labels     - Correct categories for evaluation images          10,000

print("Training images shape:", training_images.shape)
print("Training labels shape:", training_labels.shape)
print("Validation images shape:", validation_images.shape)
print("Validation labels shape:", validation_labels.shape)

# Normalize pixel values

#Each grayscale pixel has a value between 0 and 255.
#0 represents black.
#255 represents white.
#Values in between represent shades of gray.
#Dividing by 255.0 converts the range to 0–1.
#The astype("float32") converts the pixel data to 32-bit floating-point numbers.

training_images = training_images.astype("float32") / 255.0
validation_images = validation_images.astype("float32") / 255.0

# Add the channel dimension to the images.
# Each grayscale image has one channel.
# CNN input shape is (number of images, height, width, channels).

training_images = np.expand_dims(training_images, axis=-1)
validation_images = np.expand_dims(validation_images, axis=-1)

print("\nTraining images shape after adding channel:", training_images.shape)
print("Validation images shape after adding channel:", validation_images.shape)

# Define the CNN model.
# Sequential arranges the layers in the order they are executed.
# Each layer receives the output of the previous layer.

model = tf.keras.Sequential([

    # Input layer specifies the dimensions of each image.
    # Each image is 28 pixels high, 28 pixels wide, with 1 channel.

    tf.keras.layers.Input(shape=(28, 28, 1)),

    # Conv2D applies 64 filters of size 3 x 3 to the input image.
    # Filters learn image features such as edges, curves, and textures.
    # ReLU introduces non-linearity so the network can learn complex patterns.

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

    # MaxPooling2D uses a 2 x 2 window to select the maximum value.
    # This reduces the height and width of the feature maps.

    tf.keras.layers.MaxPooling2D((2, 2)),

    # The second convolutional layer learns additional image features
    # from the feature maps produced by the previous layer.

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

    # Reduce the spatial dimensions of the feature maps again.

    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten converts the feature maps into a one-dimensional vector.
    # This allows the dense layers to process the extracted features.

    tf.keras.layers.Flatten(),

    # Dense is a fully connected layer.
    # Its 128 neurons learn combinations of the extracted image features.

    tf.keras.layers.Dense(128, activation="relu"),

    # The output layer has 10 neurons, one for each clothing category.
    # Softmax converts the outputs into probabilities.
    # The category with the highest probability becomes the prediction.

    tf.keras.layers.Dense(10, activation="softmax")
])

# Display the model architecture, output shapes, and parameter counts.

model.summary()

# Compile the model to configure its training process.
# The optimizer updates the model's trainable parameters.
# The loss function measures prediction errors.
# Accuracy measures the proportion of correctly classified images.

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Stop training when validation loss stops improving.

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train the CNN with early stopping.

history = model.fit(
    training_images,
    training_labels,
    epochs=50,
    validation_data=(validation_images, validation_labels),
    callbacks=[early_stopping]
)


# Evaluate the trained model on the held-out images.
# Loss measures prediction error, while accuracy measures correct predictions.

validation_loss, validation_accuracy = model.evaluate(
    validation_images,
    validation_labels,
    verbose=2
)

print("\nTest accuracy:", validation_accuracy)
print("Test loss:", validation_loss)

# Map the integer labels from 0 to 9 to clothing category names.

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Select the first 10 held-out images for prediction.

sample_images = validation_images[:10]
sample_labels = validation_labels[:10]

# Generate predictions for the selected images.
# Each prediction contains 10 probabilities, one for each category.

predictions = model.predict(sample_images, verbose=0)

# argmax returns the index of the highest probability.
# This index represents the predicted clothing category.

predicted_labels = np.argmax(predictions, axis=1)

# Print the predicted category and the actual category for each image.

print("\nSample predictions:")

for i in range(len(sample_images)):
    predicted_name = class_names[predicted_labels[i]]
    actual_name = class_names[sample_labels[i]]

    print(
        f"Image {i + 1}: "
        f"Predicted = {predicted_name}, "
        f"Actual = {actual_name}"
    )

# Display the sample images with their predicted and actual categories.
# figsize sets the dimensions of the figure.
# subplot arranges the images in 2 rows and 5 columns.
# squeeze removes the single channel dimension for image display.

plt.figure(figsize=(14, 6))

for i in range(len(sample_images)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i].squeeze(), cmap="gray")

    predicted_name = class_names[predicted_labels[i]]
    actual_name = class_names[sample_labels[i]]

    plt.title(
        f"Predicted: {predicted_name}\nActual: {actual_name}"
    )

    # Hide axis markings to make the image grid easier to read.

    plt.axis("off")

plt.tight_layout()

# Save the image grid as a PNG file in the current working directory.
# dpi controls the resolution of the saved image.

plt.savefig(
    "fashion_mnist_predictions.png",
    dpi=200,
    bbox_inches="tight"
)

# Display the image grid.

plt.show()

# Plot the accuracy recorded during training.
# history.history stores training and validation metrics for each epoch.

plt.figure(figsize=(8, 5))

# Training accuracy shows performance on the training images.
# Validation accuracy shows performance on the held-out images.

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Test Accuracy"
)

plt.title("Fashion-MNIST CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# Display the labels identifying each line on the graph.

plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the accuracy graph as a PNG file.

plt.savefig(
    "fashion_mnist_training_accuracy.png",
    dpi=200,
    bbox_inches="tight"
)

# Display the accuracy graph.

plt.show()
