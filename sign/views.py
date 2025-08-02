# sign/views.py
import os
import numpy as np
import tempfile
import cv2
from PIL import Image
from django.shortcuts import render
from django.http import JsonResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import speech_recognition as sr
from googletrans import Translator
import sounddevice as sd
import scipy.io.wavfile as wavfile

# Load the model
try:
    model = load_model("sign_language_model.h5")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Config
img_size = (64, 64)
sequence_length = 5
base_dir = "Final"

label_dict = {
    'angry': 0, 'bye': 1, 'crying': 2, 'dance': 3, 'Deciding': 4, 'driving': 5, 'eating': 6, 'happy': 7, 'hii': 8, 'jumping': 9, 'laughing': 10, 'learning': 11, 'Planning': 12, 'playing': 13, 'please': 14, 'remembering': 15, 'running': 16, 'sad': 17, 'singing': 18, 'solving problems': 19, 'sorry': 20, 'stressed': 21, 'thinking': 22, 'walking': 23, 'welcome': 24}
inv_label_dict = {v: k for k, v in label_dict.items()}


def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            continue
        image = cv2.resize(image, img_size)
        image = img_to_array(image) / 255.0
        images.append(image)
    return np.expand_dims(np.array(images), axis=0)


def get_sign_images(text):
    folder_path = os.path.join(base_dir, text.lower())
    if not os.path.exists(folder_path):
        return None

    images = [os.path.join(folder_path, f)
              for f in sorted(os.listdir(folder_path))
              if f.lower().endswith((".jpg", "jpeg", ".png"))]

    if len(images) == 0:
        return None

    while len(images) < sequence_length:
        images.append(images[-1])

    return images[:sequence_length]


import uuid

def text_to_sign(text):
    # Load images corresponding to the spoken text
    image_paths = get_sign_images(text)
    if not image_paths or len(image_paths) < sequence_length:
        return None, "Not enough images for the detected word."

    pil_frames = [Image.open(p).convert("RGB").resize((200, 200)) for p in image_paths]

    from django.conf import settings
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

    # Generate a unique filename using UUID
    gif_filename = f"sign_result_{uuid.uuid4().hex}.gif"
    gif_path_on_disk = os.path.join(settings.MEDIA_ROOT, gif_filename)
    gif_url_for_web = settings.MEDIA_URL + gif_filename

    pil_frames[0].save(
        gif_path_on_disk,
        save_all=True,
        append_images=pil_frames[1:],
        duration=500,
        loop=0
    )

    return gif_url_for_web


def listen(request):
    try:
        recognizer = sr.Recognizer()

        # Record audio
        duration = 3
        sample_rate = 16000
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()

        # Save temp wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wavfile.write(tmp.name, sample_rate, audio_data)
            temp_wav_path = tmp.name

        # Speech Recognition
        with sr.AudioFile(temp_wav_path) as source:
            audio = recognizer.record(source)
            detected_text = recognizer.recognize_google(audio)
            cleaned_text = detected_text.strip().lower()

            if cleaned_text not in label_dict:
                return JsonResponse({"success": False, "error": "Detected word not in predefined list."})

            predicted_label = label_dict[cleaned_text]
            gif_path = text_to_sign(cleaned_text)
            
            if gif_path:
                return JsonResponse({
                    "success": True,
                    "spoken_text": detected_text,
                    "translated_text": cleaned_text,
                    "predicted_label": predicted_label,
                    "predicted_sign": cleaned_text,
                    "gif_url": gif_path
    })
            else:
                return JsonResponse({"success": False, "error": "Unable to generate sign GIF."})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})



def home(request):
    return render(request, 'sign/home.html')
