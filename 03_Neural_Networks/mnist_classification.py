import tensorflow as tf
import numpy as np


# ============================================================
# 1. LOAD THE MNIST DATASET
# ============================================================

# MNIST contains handwritten digits from 0 to 9.
#
# Training set:
# 60,000 images
#
# Validation/Test set:
# 10,000 images

data = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()


# ============================================================
# 2. NORMALIZE THE IMAGE DATA
# ============================================================

# Each MNIST pixel has a value between 0 and 255.
#
# Dividing by 255 converts the values to the range:
#
# 0.0 -> black
# 1.0 -> white

training_images = training_images / 255.0
test_images = test_images / 255.0


# ============================================================
# 3. DEFINE THE NEURAL NETWORK
# ============================================================

model = tf.keras.models.Sequential([
    
    # Convert each 28x28 image into a 1D vector
    # containing 784 values.
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # Hidden layer containing 20 neurons.
    # ReLU is used as the activation function.
    tf.keras.layers.Dense(
        20,
        activation=tf.nn.relu
    ),

    # Output layer containing 10 neurons.
    # One neuron represents each digit: 0 through 9.
    #
    # Softmax converts the outputs into probabilities.
    tf.keras.layers.Dense(
        10,
        activation=tf.nn.softmax
    )
])


# ============================================================
# 4. COMPILE THE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 5. TRAIN THE MODEL
# ============================================================

model.fit(
    training_images,
    training_labels,
    validation_data=(val_images, val_labels)
    epochs=20
)


# ============================================================
# 6. EVALUATE THE MODEL
# ============================================================

test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels,
    verbose=2
)

print("\nTest accuracy:", test_accuracy)
print("Test loss:", test_loss)


# ============================================================
# 7. MAKE PREDICTIONS
# ============================================================

classifications = model.predict(test_images)

# Display the probability distribution
# for the first test image.
print("\nPrediction probabilities for first image:")
print(classifications[0])


# ============================================================
# 8. FIND THE PREDICTED DIGIT
# ============================================================

predicted_digit = np.argmax(classifications[0])

print("\nPredicted digit:", predicted_digit)
print("Actual digit:", test_labels[0])


# ============================================================
# 9. DISPLAY THE CONFIDENCE
# ============================================================

confidence = classifications[0][predicted_digit]

print("Confidence:", confidence)
print("Confidence percentage:", confidence * 100, "%")