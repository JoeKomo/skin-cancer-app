# main.py

import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.metrics import top_k_categorical_accuracy
from util import classify

# Define top_2_accuracy and top_3_accuracy before loading the model
def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

# set title
st.title('Skin Cancer Prediction')

# set header
st.header('Please upload a skin cancer image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/model.h5', custom_objects={'top_2_accuracy': top_2_accuracy, 'top_3_accuracy': top_3_accuracy})

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name_results, conf_score_results = classify(image, model, class_names)

    # Display the top 3 predictions
    st.markdown("<p style='font-size: 24px; font-weight: bold;'>Result:</p>", unsafe_allow_html=True)
    
    # Sort the predictions by confidence score in descending order
    sorted_predictions = sorted(zip(class_name_results, conf_score_results), key=lambda x: x[1], reverse=True)

    # Display the top 3 predictions
    for i, (class_name, conf_score) in enumerate(sorted_predictions[:3]):
        st.write(f"{i + 1}. {class_name}: {conf_score * 100:.1f}%")
