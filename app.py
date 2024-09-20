import numpy as np
import io
import os
from PIL import Image
from flask import request, jsonify, Flask
from flask_cors import CORS

import tensorflow as tf

from tensorflow.keras.models import Model,load_model
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Flatten, Reshape, Dense, Concatenate, Dropout, LSTM,GlobalAveragePooling2D,BatchNormalization


import cv2
import pandas as pd
from PIL import Image
from ultralytics import YOLO


app = Flask(__name__)
CORS(app)

# Global variables to hold models, will be lazily loaded
cnn_model = None
yolo_model = None

def create_vgg19_model():
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(255, 255, 3))
    base_model.trainable = False  # Freeze the base model to prevent updating its weights
    return base_model

# Define your LSTM model
def create_lstm_model():
    lstm_model = tf.keras.Sequential([
        LSTM(units=64, return_sequences=True),
        LSTM(units=32)
    ])
    return lstm_model

def create_ensemble_model():
    vgg19_model = create_vgg19_model()  # Using VGG19 as CNN

    lstm_model = create_lstm_model()  # Your LSTM model

    cnn_output = vgg19_model.output
    cnn_output_flat = Flatten()(cnn_output)  # Flatten the CNN output
    cnn_output_reshaped = Reshape((1, -1))(cnn_output_flat)  # Reshape to match LSTM input shape

    lstm_output = lstm_model(cnn_output_reshaped)  # Pass reshaped CNN output to LSTM

    # Concatenate the outputs
    merged_output = Concatenate()([cnn_output_flat, lstm_output])

    # Add a dense layer for classification (adjust the number of classes according to your dataset)
    output_layer = Dense(8, activation='softmax')(merged_output)

    # Create the ensemble model
    ensemble_model = Model(inputs=vgg19_model.input, outputs=output_layer)
    ensemble_model.compile(optimizer='Adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    return ensemble_model


#Loading Model
def load_models():
    global cnn_model, yolo_model
    
    if cnn_model is None:
        cnn_model = create_ensemble_model()
        cnn_model.load_weights('model/sea_cucumber_system_final_v2.h5')
        #cnn_model = load_model('model/sea_cucumber_system_final_v2.h5')

    if yolo_model is None:
        yolo_model = YOLO('model/best.pt')
    
    print(" * Models loaded successfully")
    return cnn_model,yolo_model

index = ['color', 'color_name', 'hex', 'R', 'G', 'B']
df = pd.read_csv("colors.csv", header=None, names=index)

def colorname(B,G,R):
    minimum = 10000
    for i in range(len(df)):
        d = abs(B-int(df.loc[i,"B"])) + abs(G-int(df.loc[i,"G"])) + abs(R-int(df.loc[i,"R"]))
        if (d<=minimum):
            minimum = d
            cname = df.loc[i,"color_name"] + "Hex" + df.loc[i, "hex"]
    return cname

def find_image_center(image):
    height, width = image.shape[:2]
    center_x = width / 2
    center_y = height / 2
    return center_x, center_y

def prediction_probability_label(model,modelYolo, image, class_labels, is_rgb=True)->tuple:
   
   # Convert the PIL image to an array and resize it
    if is_rgb:
        image = image.convert("RGB")
        img = image.resize((255, 255))  # Resize the image
    else:
        image = image.convert("L")
        img = image.resize((255, 255))

    # Convert image to numpy array
    input_arr = np.array(img)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch.
    input_arr = input_arr / 255.0  # Normalize the image

    # Predict using the CNN+LSTM model
    pred_probs = model.predict(input_arr)[0]
    pred_class = np.argmax(pred_probs)
    pred_label = class_labels[pred_class]
    pred_prob = round(pred_probs[pred_class] * 100, 2)

    # Convert the original image to a numpy array for YOLO model processing
    img_array = np.array(image)
    center_x, center_y = find_image_center(img_array)
    print("Center of the image:", center_x, ",", center_y)
    b, g, r = img_array[int(center_y), int(center_x)]
    text = colorname(b, g, r) + '   R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)

    # Run inference using the YOLO model
    results = modelYolo(img_array)  # results list

    # View results
    for r in results:
        boxes = r.boxes
        masks  = r.masks
        x, y, w, h = 0,0,0,0
        x1, y1, x2, y2 = 0,0,0,0

        for box in boxes:
        # if boxes:
            box = boxes[0]
            print(box.xywh[0])
            x, y, w, h = map(int,box.xywh[0])
            x1, y1, x2, y2 = map(int,box.xyxy[0])
        print(x, y, w, h)
        print( x1, y1, x2, y2 )

        img_array = np.array(image)
        img_cropped = img_array[y1:y2, x1:x2]


        # Example usage:
        # Assuming 'image' is your numpy array representing the image
        center_x, center_y = find_image_center(img_cropped)
        print("Center of the image:", center_x, ",", center_y)
        b,g,r = img_cropped[int(center_y),int(center_x)]
        text = colorname(b,g,r) + '   R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)

        print("Object width:", w*0.26," mm")
        print("Object height:", h*0.26," mm")

        # image = Image.open(img_path)
        # resolution_x, resolution_y = image.info['dpi']

        try:
            resolution_x, resolution_y = image.info['dpi']
        except KeyError:
            # Default to 96 DPI if DPI info is not present
            resolution_x, resolution_y = 96, 96

        # Convert DPI to PPI (pixels per inch)
        ppi_x = resolution_x / 25.4  # Convert DPI to PPI
        ppi_y = resolution_y / 25.4

        # Calculate physical size of each pixel (in mm)
        pixel_size_x = 25.4 / ppi_x
        pixel_size_y = 25.4 / ppi_y

        # Convert pixels to millimeters
        width_mm_pixels =  w*25.4 /resolution_x
        height_mm_pixels = h*25.4/resolution_y

        # Density_Factors
        # Assuming a density factor to convert from weight/height per pixel to actual weight/height
        # Adjust this value based on your specific application

        if pred_label =='Bohadschia Marmorata Class A':
            density_factor = 0.15

        elif pred_label =='Bohadschia Vitiensis Class A':
            density_factor = 0.14

        elif pred_label =='Holothuria Spinifera Class A':
            density_factor = 0.15

        elif pred_label =='Holothuria Spinifera Class B':
            density_factor = 0.15

        elif pred_label =='Stichopus Naso Class A':
            density_factor = 0.13
        
        elif pred_label =='Holothuria Scabra Class A':
            density_factor = 0.18
        
        elif pred_label =='Holothuria Scabra Class B':
            density_factor = 0.20
        
        elif pred_label =='Holothuria Scabra Class C':
            density_factor = 0.15
        
        else:
            density_factor = 0.18

        #Convert pixel values to weight and height
        width_mm = width_mm_pixels * density_factor
        height_mm = height_mm_pixels * density_factor

        print(f"Width: {width_mm} mm")
        print(f"Height: {height_mm} mm")
        print()

    return (pred_label, pred_prob,text)

@app.route("/predict", methods=["POST"])
def predict():
    try: 
        # Ensure an image file is present in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file in the request'}), 400

        # Retrieve the image file
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Open the image file
        image = Image.open(file.stream)
        print(f' * Image: {image}')

        img_path = "C:/Users/Jeevake/Seacucumber/2024_09_15/demarcations/Bohadschia Marmorata Class A/PXL_20240408_051936593.jpg"
        class_labels = ['Bohadschia Marmorata Class A', 'Bohadschia Vitiensis Class A', 'Holothuria Spinifera Class A', 'Holothuria Spinifera Class B', 'Stichopus Naso Class A','Holothuria Scabra Class C','Holothuria Scabra Class A','Holothuria Scabra Class B']

        model,yolo_model = load_models()
        pred_label,pred_prob,text = prediction_probability_label(model,yolo_model, image, class_labels)

        # print(f'Predicted label for the image: {pred_label}')
        # print(f'Confidence level: {pred_prob}')
        # print(f'Text: {text}')
        
        response = {
            'prediction': {
                'pred_label': pred_label,
                'pred_prob': pred_prob,
                'text': text,
            }
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
