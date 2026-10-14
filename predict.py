
import tensorflow as tf

# Load the saved model from the current folder
model = tf.keras.models.load_model("mnist_cnn.keras")

print("Model loaded successfully!")
