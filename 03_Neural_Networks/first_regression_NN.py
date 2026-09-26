import numpy as np
import tensorflow as tf

# Training data 
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Create the neural network
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

# Configure training 
model.compile(optimizer = "sgd", loss = "mean_squared_error")

# Train the model
model.fit(xs, ys, epochs=500)

# Test the trained model
prediction = model.predict(np.array([10.0]))
print("prediction for x = 10", prediction)
