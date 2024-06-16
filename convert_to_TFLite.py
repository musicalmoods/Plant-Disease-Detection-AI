import tensorflow as tf

# Load your model
model = tf.keras.models.load_model('my_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with tf.io.gfile.GFile('my_model.tflite', 'wb') as f:
    f.write(tflite_model)
