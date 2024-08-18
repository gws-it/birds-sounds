import librosa
import numpy as np
import sklearn.preprocessing
from PIL import Image
from uuid import uuid4
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings("ignore")

# Load the full model
model = load_model('best_model.keras')


classes_to_predict=['Aegithina tiphia',
 'Ardea alba',
 'Ardea cinerea',
 'Ardea purpurea',
 'Arenaria interpres',
 'Corvus macrorhynchos',
 'Dicrurus paradiseus',
 'Elanus caeruleus',
 'Eudynamys scolopaceus',
 'Gallinula chloropus',
 'Motacilla cinerea',
 'Orthotomus sutorius',
 'Passer domesticus',
 'Psittacula krameri',
 'Tyto alba']

def Prediction_bird(filename):
    wave_data, wave_rate = librosa.load(filename)
    wave_data, _ = librosa.effects.trim(wave_data)
    sample_length = 5 * wave_rate
    N_mels = 216
    for idx in range(0, len(wave_data), sample_length):
        song_sample = wave_data[idx:idx+sample_length]
        if len(song_sample) >= sample_length:
            mel = librosa.feature.melspectrogram(y=song_sample, sr=wave_rate, n_mels=N_mels)
            db = librosa.power_to_db(mel)
            normalised_db = sklearn.preprocessing.minmax_scale(db)
            db_array = (np.asarray(normalised_db) * 255).astype(np.uint8)
            db_image = Image.fromarray(np.array([db_array, db_array, db_array]).T)
            return db_image

def preprocess_image(image):
    image = image.resize((216, 216))
    image_array = np.array(image)
    image_array = image_array / 255.0  # Rescale to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Make predictions on new data
# new_predictions = model.predict(new_audio_data)
# Predict
bird_image = Prediction_bird("temp_audio_file.wav")
bird_image_array = preprocess_image(bird_image)
prediction = model.predict(bird_image_array)
predicted_class = classes_to_predict[np.argmax(prediction)]

print(f"Predicted class: {predicted_class}")