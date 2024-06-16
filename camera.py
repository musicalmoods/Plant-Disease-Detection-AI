import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# Load the model
model = load_model('my_model.h5')

# Start the webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the image for prediction
    img = cv2.resize(frame, (150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Use the model to make a prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    # Map the class index to its corresponding label
    labels = ['Bacterial wilt disease', 'Healthy', 'Manganese Toxicity']
    predicted_label = labels[class_index]

    # Display the resulting frame with the prediction
    cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Plant Disease Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
