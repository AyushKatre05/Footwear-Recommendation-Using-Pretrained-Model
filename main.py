import os
import cv2
import numpy as np
from tqdm import tqdm
import streamlit as st
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from sklearn.neighbors import NearestNeighbors

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Define the extract_feature function
def extract_feature(img_path, model):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to read image '{img_path}'. Skipping...")
            return None
        img = cv2.resize(img, (224, 224))  # Resize to match ResNet50 input size
        img = np.array(img)
        expand_img = np.expand_dims(img, axis=0)
        pre_img = preprocess_input(expand_img)
        result = model.predict(pre_img).flatten()
        normalized = result / np.linalg.norm(result)  # Use np.linalg.norm for calculating norm
        return normalized
    except Exception as e:
        print(f"Error processing '{img_path}': {e}")
        return None

# Define data directories for different categories
men_directory = r'C:\Users\ayush\OneDrive\Desktop\Footwear Recommendation System\Footwear\Men\Images\images_with_product_ids'
women_directory = r'C:\Users\ayush\OneDrive\Desktop\Footwear Recommendation System\Footwear\Women\Images\images_with_product_ids'

# List of directories containing images
data_directories = {'men': men_directory, 'women': women_directory}

# Collect paths of all images in the specified directories for each category
filenames = {}
for category, directory in data_directories.items():
    filenames[category] = []
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            filenames[category].append(full_path)

# Streamlit app
st.title('ShopSnapster - Shoe Recommendation System')

# File upload section
uploaded_file = st.file_uploader("Upload an image of the shoe you like", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        resized_img = display_image.resize((224, 224))  # Resize image to match model input shape
        st.image(resized_img)

        # Feature extraction
        img_array = np.array(resized_img)
        preprocessed_img = preprocess_input(img_array)
        features = model.predict(np.expand_dims(preprocessed_img, axis=0)).flatten()

        # Recommendation for men's shoes
        if st.checkbox("Recommend for Men"):
            men_features = pickle.load(open('men_features.pkl', 'rb'))
            men_filenames = pickle.load(open('men_filenames.pkl', 'rb'))
            nn_model = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
            nn_model.fit(men_features)
            distances, indices = nn_model.kneighbors([features])
            st.subheader("Recommended Men's Shoes:")
            for index in indices[0][1:]:
                st.image(Image.open(men_filenames[index]))

        # Recommendation for women's shoes
        if st.checkbox("Recommend for Women"):
            women_features = pickle.load(open('women_features.pkl', 'rb'))
            women_filenames = pickle.load(open('women_filenames.pkl', 'rb'))
            nn_model = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
            nn_model.fit(women_features)
            distances, indices = nn_model.kneighbors([features])
            st.subheader("Recommended Women's Shoes:")
            for index in indices[0][1:]:
                st.image(Image.open(women_filenames[index]))

    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
