import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import base64
import subprocess
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import base64
import librosa
import numpy as np
import sklearn.preprocessing
from PIL import Image
from uuid import uuid4
import tensorflow as tf
import keras
# from tensorflow.keras.models import load_model
import json

# # Load the full model
# model = load_model('best_model.keras')
# # model.load_weights("model.weights.h5")


# classes_to_predict=['Aegithina tiphia',
#  'Ardea alba',
#  'Ardea cinerea',
#  'Ardea purpurea',
#  'Arenaria interpres',
#  'Corvus macrorhynchos',
#  'Dicrurus paradiseus',
#  'Elanus caeruleus',
#  'Eudynamys scolopaceus',
#  'Gallinula chloropus',
#  'Motacilla cinerea',
#  'Orthotomus sutorius',
#  'Passer domesticus',
#  'Psittacula krameri',
#  'Tyto alba']

# def Prediction_bird(filename):
#     wave_data, wave_rate = librosa.load(filename)
#     wave_data, _ = librosa.effects.trim(wave_data)
#     sample_length = 5 * wave_rate
#     N_mels = 216
#     for idx in range(0, len(wave_data), sample_length):
#         song_sample = wave_data[idx:idx+sample_length]
#         if len(song_sample) >= sample_length:
#             mel = librosa.feature.melspectrogram(y=song_sample, sr=wave_rate, n_mels=N_mels)
#             db = librosa.power_to_db(mel)
#             normalised_db = sklearn.preprocessing.minmax_scale(db)
#             db_array = (np.asarray(normalised_db) * 255).astype(np.uint8)
#             db_image = Image.fromarray(np.array([db_array, db_array, db_array]).T)
#             return db_image

# def preprocess_image(image):
#     image = image.resize((216, 216))
#     image_array = np.array(image)
#     image_array = image_array / 255.0  # Rescale to [0, 1]
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#     return image_array

def prediction(audio_file):

    # Load the Prediction JSON File to Predict Target_Label
    with open('prediction.json', mode='r') as f:
        prediction_dict = json.load(f)

    # Extract the Audio_Signal and Sample_Rate from Input Audio
    audio, sample_rate =librosa.load(audio_file)

    # Extract the MFCC Features and Aggrigate
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_features = np.mean(mfccs_features, axis=1)

    # Reshape MFCC features to match the expected input shape for Conv1D both batch & feature dimension
    mfccs_features = np.expand_dims(mfccs_features, axis=0)
    mfccs_features = np.expand_dims(mfccs_features, axis=2)

    # Convert into Tensors
    mfccs_tensors = tf.convert_to_tensor(mfccs_features, dtype=tf.float32)

    # Load the Model and Prediction
    model = tf.keras.models.load_model('model.h5')
    prediction = model.predict(mfccs_tensors)

    # Find the Maximum Probability Value
    target_label = np.argmax(prediction)

    # Find the Target_Label Name using Prediction_dict
    predicted_class = prediction_dict[str(target_label)]
    confidence = round(np.max(prediction)*100, 2)
    return predicted_class,confidence

    # print(f'Predicted Class : {predicted_class}')
    # print(f'Confident : {confidence}%')


# Function to read a local image file and convert it to a base64 string
@st.cache_data
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to convert image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
    
# Load the background image
bg_img_path = "stat_imgs/design.jpg"
bg_img_base64 = get_img_as_base64(bg_img_path)

# Load the sidebar image
sidebar_img_path = "GWSLivingArt_Logo.png"
sidebar_img_base64 = get_img_as_base64(sidebar_img_path)

# Load company logo
logo_image_path = '\GWSLivingArt_Logo.png'
logo_image = Image.open(logo_image_path)

# Convert the logo image to a base64 string
logo_base64 = image_to_base64(logo_image)

# Set the background image and sidebar image in the Streamlit app
# Define custom CSS for styling
page_bg_img = f"""
<style>
/* Top banner styling */
.top-banner {{
    width: 100%;
    height: 80px;
    background-color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    # box-shadow: 0px 4px 2px -2px gray;
    position: fixed;
    top: 0;
    left: 0;
    z-index: 10000;
}}

/* Styling for the logo in the banner */
.top-banner-logo {{
    height: 100%; /* Makes the logo height match the banner */
    max-height: 80px; /* Limit maximum height of the logo */
    width: auto; /* Maintain aspect ratio */
}}

/* Styling for the text in the banner */
.top-banner-text {{
    font-family: "Heebo", sans-serif;
    font-size: 24px;
    color: #09B37A; /* Company theme color */
    margin-left: 20px;
}}

/* Background and Sidebar Image */
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{bg_img_base64}");
    background-size: 100%;
    background-position: top left;
    background-repeat: repeat;
    background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{sidebar_img_base64}");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

# [data-testid="stHeader"] > div:first-child {{
#     display: flex;
#     align-items: center;
#     justify-content: center;
# }}

.header-logo {{
    position: fixed;
    top: 0px;
    left: 0px;
    padding: 5px;
    z-index: 10000;
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}

/* Styling for buttons */
button {{
    background-color: #09B37A !important;
    color: white !important;
    border-radius: 5px !important;
    border: none !important;
}}

button:hover {{
    background-color: #07A06A !important;
    color: white !important;
}}

.st-emotion-cache-h4xjwg {{
    position: fixed;
    top: 0px;
    left: 0px;
    right: 0px;
    height: 0rem;
    background: rgb(14, 17, 23);
    outline: none;
    z-index: 999990;
    display: block;
}}

/* Styling for filenames and text */
.st-emotion-cache-7oyrr6 {{
    color: #09B37A !important;
    font-size: 16px !important;
    line-height: 1.5 !important;
}}

.st-emotion-cache-1uixxvy {{
    color: #09B37A !important;
    margin-right: 0.5rem !important;
    margin-bottom: 0.25rem !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}}

.st-emotion-cache-13ln4jf {{
    max-width: 50rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}}

p {{
    color: white;
    font-family: "Heebo", sans-serif !important;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# HTML for the top banner
top_banner_html = f"""
<div class="top-banner">
    <a href="https://www.gwslivingart.com" target="_blank">
        <img src="data:image/png;base64,{logo_base64}" class="top-banner-logo">        
    </a>
    <div class="top-banner-text"></div>
</div>
"""


# Inject the HTML into the Streamlit app
st.markdown(top_banner_html, unsafe_allow_html=True)

# Adjust padding for main content to avoid overlap with the banner
st.markdown("<div style='padding-top: 100px;'></div>", unsafe_allow_html=True)


# st.image("title.jpg")
df=pd.read_csv("Birds_full_data.csv")


# Function to display the waveform
def display_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title='Waveform')
    st.pyplot(fig)

# Function to get bird description using OpenAI API
def get_bird_description(bird_name):
    return df[df.common_name==f"{bird_name}"].description.values[0]
 

# Streamlit UI
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;700&display=swap" rel="stylesheet">
    <h1 style='text-align: center; color: #09B37A; font-family: "Heebo",'>Birds Sound Identification System</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;700&display=swap" rel="stylesheet">
    <h3 style='text-align: center; color: #09B37A; font-family: "Heebo",'>Upload Your Sound Clip Here</h3>
    """,
    unsafe_allow_html=True
)


# Upload audio file
uploaded_file = st.file_uploader("", type=["wav", "mp3","ogg"])

if uploaded_file is not None:
    # Load audio file
    audio_data, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file, format='audio/wav')

    # Display waveform
    # st.subheader('Original Sound')
    st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;700&display=swap" rel="stylesheet">
    <h6 style='text-align: left; color: #09B37A; font-family: "Heebo",'>Original Sound</h6>
    """,
    unsafe_allow_html=True
)
    display_waveform(audio_data, sr)

    # Predict bird species button
    if st.button('Predict Bird Species'):
        predicted_class,confidence = prediction(uploaded_file)
        if confidence >70:

            bird_species = df[df.scientific_name==f"{predicted_class}"].common_name.values[0]#"Crested Honey Buzzard"
            st.markdown(
                f"""
                <link href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;700&display=swap" rel="stylesheet">
                <h2 style='color: #09B37A; font-family: "Heebo",'>Predicted Bird Species: {bird_species}</h2>
                """,
                unsafe_allow_html=True)
            st.markdown(
                f"""
                <link href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;700&display=swap" rel="stylesheet">
                <h6 style='color: #09B37A; font-family: "Heebo",'> Model Confidence: {confidence} %</h6>
                """,
                unsafe_allow_html=True)
            

    
            ##here need to update the image display

            # Display bird image (assuming you have images named as the bird species)
            image_formats = ['jpg', 'png']
            image_found = False
            for ext in image_formats:
                bird_image_path = f'images2/{bird_species}.{ext}'
                if os.path.isfile(bird_image_path):

                    bird_image = Image.open(bird_image_path)
                    bird_image_base64 = logo_base64 = image_to_base64(bird_image)
                    
                    # Display the image using HTML and CSS to center it
                    st.markdown(
                        f'''
                        <div style="display: flex; justify-content: center;">
                            <img src="data:image/png;base64,{bird_image_base64}" width="300">
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
                    image_found = True
                    break
            
            if not image_found:
                st.write("No image available for this bird species.")

            # Get and display bird description
            description = get_bird_description(bird_species)
            st.write(" ")
            st.markdown(f"""
                        <p style='color: White; font-family: "Heebo",'>{description}</p>
                        """,
                        unsafe_allow_html=True)
        else:
            st.markdown(
    """
     <link href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;700&display=swap" rel="stylesheet">
    <h3 style='text-align: center; color: #09B37A; font-family: "Heebo",'>❌ This sound clip is Not Available in the Trained Data</h3>
    """,
    unsafe_allow_html=True
)


    
    
    

        
# Load company logo
logo_image = Image.open('GWSLivingArt_Logo_V6-02.png')

# Convert the logo image to a base64 string
logo_base64 = image_to_base64(logo_image)

# Display the company logo in small size and centered
st.markdown(
    f'''
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{logo_base64}" width="250">
    </div>
    ''',
    unsafe_allow_html=True
)
