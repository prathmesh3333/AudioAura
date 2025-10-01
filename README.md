AudioAura is an AI-driven speech emotion recognition system capable of detecting multiple emotions (happy, sad, angry, neutral, etc.) from audio inputs in real time. This repository provides a Google Colab notebook to run the system, along with instructions to load the pretrained model stored on Google Drive.

Project Structure

AudioAura_Colab.ipynb – Google Colab notebook to run the system.

model/ – Folder to store the extracted pretrained model (from Google Drive).

Requirements

Python 3.x

transformers library

torch

librosa

Google Colab environment (recommended)

How to Run

Mount your Google Drive in Colab
This is required to access the pretrained model stored on your Drive.

from google.colab import drive
drive.mount('/content/drive')


Set the model path and extract the model

import zipfile

# Path to the model zip file on your Google Drive
model_path = "/content/drive/My Drive/Audio_Aura_Model/audio_aura_model.zip"

# Destination folder to extract the model
destination = "/content/model"

# Extract the model
with zipfile.ZipFile(model_path, 'r') as zip_ref:
    zip_ref.extractall(destination)

print("Model extracted successfully!")


Load the pretrained model and processor

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# Load the model and processor
model = Wav2Vec2ForSequenceClassification.from_pretrained(destination)
processor = Wav2Vec2Processor.from_pretrained(destination)

print("Model loaded successfully!")


Start using the system
Once the model is loaded, you can pass audio files to the system to predict emotions. Example usage is provided in the Colab notebook.
