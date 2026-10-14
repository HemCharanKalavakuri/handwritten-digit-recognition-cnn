import os
os.environ["KERAS_HOME"]=r"C:\AI_Projects\keras_data"
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)

print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)

import matplotlib.pyplot as plt

# Display the first training image
plt.imshow(x_train[0], cmap="gray")

# Display the correct digit
plt.title(f"Actual Digit: {y_train[0]}")

# Hide the coordinate axes
plt.axis("off")

# Show the image
# plt.show()

print("Image shape:", x_train[0].shape)
print("Pixel values:")
print(x_train[0])

print("Actual label:", y_train[0])

# Normalize the pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())

# the tf.newaxis whisch is used a new axis to the shape before our x_tain shape is (60000,28,28) but the cnn layer need the 4 dimension fot the grayscale image so this tf.newaxis add the aditional dimension 
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print(x_train.shape)
print(x_train.shape)


# Create a CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32, kernel_size=(3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dense(10, activation="softmax")
])

# Display the model architecture
model.summary()


# compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# trainning
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)


import matplotlib.pyplot as plt

training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]

epochs = range(1, len(training_loss) + 1)

# Create a new figure for the line graph
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(epochs, training_loss, color="blue",
        marker="o", linestyle="-", label="Training Loss")

ax.plot(epochs, validation_loss, color="orange",
        marker="o", linestyle="-", label="Validation Loss")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training vs Validation Loss")
ax.set_xticks(list(epochs))
ax.legend()
ax.grid(True)

plt.show()

# Evaluate the model on unseen test images
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")

import numpy as np
index=0
image=x_train[index]

prediction=model.predict(np.expand_dims(image,axis=0),verbose=0)

predicted_digit=np.argmax(prediction[0])

actual_digit=y_train[index]
plt.imshow(image,cmap="gray")
plt.title(
    f"Predicted: {predicted_digit} | Actual: {actual_digit}"
)
plt.axis("off")
plt.show()

print("Predicted digit:", predicted_digit)
print("Actual digit:", actual_digit)
print(
    "Confidence:",
    round(float(np.max(prediction[0])) * 100, 2),
    "%"
)

# Save the trained CNN model
model.save("mnist_cnn.keras")

print("Model saved successfully!")
