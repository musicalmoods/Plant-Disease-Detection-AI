from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

# Load the model
model = load_model('my_model.h5')

# Load the image you want to classify
image_path = input("Enter the path where the image to be predicted is located: ")
img = image.load_img(image_path, target_size=(150, 150))

# Convert the image to a numpy array
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Normalize the image
img_array /= 255.

# Use the model to make a prediction
prediction = model.predict(img_array)
class_index = np.argmax(prediction)

# Map the class index to its corresponding label
labels = ['Bacterial wilt disease', 'Healthy', 'Manganese Toxicity']
predicted_label = labels[class_index]

print(f'The model predicts that the image is: {predicted_label}')
