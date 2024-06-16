import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2 #type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from PIL import Image
from pathlib import Path

# Load the dataset
df = pd.read_csv(r"X:\\Musical_Moods\\Data\\Plant_Disease_Data\\get_leaves\\train.csv")

# Preprocess the bounding box data
df['bbox'] = df['bbox'].apply(lambda x: np.array([int(num) for num in x.strip('[]').split(',')], dtype='float32'))

# Split the dataset
train_df, valid_df = train_test_split(df, test_size=0.2)

# Define the base pre-trained model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# And a logistic layer with 4 classes (for 4 bounding box coordinates)
predictions = Dense(4, activation='linear')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First, we only train the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model (should be done after setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.001), loss='mse')

# Manually preprocess the images and bounding boxes
def preprocess_images_and_bbox(df, img_folder):
    img_data = []
    bbox_data = []
    for _, row in df.iterrows():
        # Open and resize the image
        img = Image.open(os.path.join(img_folder, row['image_id']))
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img_data.append(img)

        # Get the bounding box
        bbox = row['bbox']
        bbox_data.append(bbox)

    return np.array(img_data), np.array(bbox_data)

# Preprocess the training and validation data
train_images, train_bbox = preprocess_images_and_bbox(train_df, "X:\\Shlok_Shah\\Data\\Plant_Disease_Data\\get_leaves\\train")
valid_images, valid_bbox = preprocess_images_and_bbox(valid_df, "X:\\Shlok_Shah\\Data\\Plant_Disease_Data\\get_leaves\\test\\leaf")

# Fit the model
model.fit(x=train_images, y=train_bbox, validation_data=(valid_images, valid_bbox), epochs=100)

# Save the model
model.save('leaf_getter.h5')
