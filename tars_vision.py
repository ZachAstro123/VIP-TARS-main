import pygame
import os
import numpy as np
import requests
import cv2
import base64
from pyscreenshot import grab
import io
from pathlib import Path
import threading
import mss
import sys
import openai
import elevenlabs
from typing import Dict, List, Tuple
import faiss
import pickle
import json
import face_recognition
import concurrent.futures
import soundfile as sf
import sounddevice as sd
import asyncio
import websocket
import pyaudio
import time
from ibm_watson import DiscoveryV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import queue
import random
from pydub import AudioSegment
from pydub.playback import play
#from pyht import Client
#from dotenv import load_dotenv
#from pyht.client import TTSOptions
import pygame

# Set environment variable to suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
pygame.init()


print("Waking up TARS...")

api_key = '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'
elevenlabs.set_api_key("")
elevenlabs_voice_id = ''  # female voice: 21m00Tcm4TlvDq8ikWAM

class asyncIBM:

    def __init__(self, TARS):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.TARS = TARS
        self.texts = TARS.transcript
        self.audio_data_buffer = b""
        self.websocket_open = False
        self.queue_of_texts = queue.Queue()
        self.ws_url = 'wss://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/a060b1a1-8b30-4191-b335-095ddbf0c1ad/v1/recognize'
        self.another_token = 'eyJraWQiOiIyMDIzMTEwNzA4MzYiLCJhbGciOiJSUzI1NiJ9.eyJpYW1faWQiOiJpYW0tU2VydmljZUlkLWU2YmM1ZTQ1LTcyYWYtNDMzOS1hZWYxLTFjMjRjYzRmMjUwZiIsImlkIjoiaWFtLVNlcnZpY2VJZC1lNmJjNWU0NS03MmFmLTQzMzktYWVmMS0xYzI0Y2M0ZjI1MGYiLCJyZWFsbWlkIjoiaWFtIiwianRpIjoiMWY4MzUwNDQtZDA2OC00OTkxLTkzZmMtMDkyMjUyMDYyYzY3IiwiaWRlbnRpZmllciI6IlNlcnZpY2VJZC1lNmJjNWU0NS03MmFmLTQzMzktYWVmMS0xYzI0Y2M0ZjI1MGYiLCJuYW1lIjoiQXV0by1nZW5lcmF0ZWQgc2VydmljZSBjcmVkZW50aWFscyIsInN1YiI6IlNlcnZpY2VJZC1lNmJjNWU0NS03MmFmLTQzMzktYWVmMS0xYzI0Y2M0ZjI1MGYiLCJ1bmlxdWVfaW5zdGFuY2VfY3JucyI6WyJjcm46djE6Ymx1ZW1peDpwdWJsaWM6c3BlZWNoLXRvLXRleHQ6YXUtc3lkOmEvYjE2YzU3ZTMyMWU1NGI1YmI1YmFlZGUzMTc5M2RhMTg6YTA2MGIxYTEtOGIzMC00MTkxLWIzMzUtMDk1ZGRiZjBjMWFkOjoiXSwic3ViX3R5cGUiOiJTZXJ2aWNlSWQiLCJhdXRobiI6eyJzdWIiOiJTZXJ2aWNlSWQtZTZiYzVlNDUtNzJhZi00MzM5LWFlZjEtMWMyNGNjNGYyNTBmIiwiaWFtX2lkIjoiaWFtLVNlcnZpY2VJZC1lNmJjNWU0NS03MmFmLTQzMzktYWVmMS0xYzI0Y2M0ZjI1MGYiLCJzdWJfdHlwZSI6IlNlcnZpY2VJZCIsIm5hbWUiOiJBdXRvLWdlbmVyYXRlZCBzZXJ2aWNlIGNyZWRlbnRpYWxzIn0sImFjY291bnQiOnsidmFsaWQiOnRydWUsImJzcyI6ImIxNmM1N2UzMjFlNTRiNWJiNWJhZWRlMzE3OTNkYTE4IiwiZnJvemVuIjp0cnVlfSwiaWF0IjoxNzAxNzQwMTg5LCJleHAiOjE3MDE3NDM3ODksImlzcyI6Imh0dHBzOi8vaWFtLmNsb3VkLmlibS5jb20vaWRlbnRpdHkiLCJncmFudF90eXBlIjoidXJuOmlibTpwYXJhbXM6b2F1dGg6Z3JhbnQtdHlwZTphcGlrZXkiLCJzY29wZSI6ImlibSBvcGVuaWQiLCJjbGllbnRfaWQiOiJkZWZhdWx0IiwiYWNyIjoxLCJhbXIiOlsicHdkIl19.BKWyfOFJj-fbjuaZPZmrNf8N3pAGBbjVClNprMZXv6EYiBG63CnpjEITuZsy0jUWVdhvzm7qKQQN_uskQ9RfNTV2tdwl27mYTtGEs9GcucPAS-HWer0uwvc6oAMqKjQ00tbZ7nBt04X-3xcVjwENDKoPxXG5p5ksvf-NIAwvyx7Z0-2nEx6j-hmBN0rLa1mb1iy-5yjwdP-mXqflydFeNp9nHpTIwri_Q0Oy7aaPAestQpogItWWKU0Ij0p2-jgFzowyCAVz0AQRq8TyInw11cducZ7oPg9wP3CBOYXvixr3KHxAFrDcLBlPWhjfM4EFYSW2dGvHZPqvL81nvv79ow'
        self.wsURI = self.ws_url + '/v1/recognize?access_token=' + \
            self.another_token + '&model=en-US_BroadbandModel'
        self.p = pyaudio.PyAudio()
        #curl -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=CA6nqYYeVitl4mvomK8U704oEp_NhLR4EyUt6tI_udkR" "https://iam.cloud.ibm.com/identity/token"

        self.ws = websocket.WebSocketApp(self.wsURI, on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK,
                                  stream_callback=self.audio_stream_callback)
        self.string = ''

    def on_message(self, ws, message):
        result = json.loads(message)
        if "results" in result:
            if not self.TARS.user_speaking:
                self.TARS.user_speaking = True
            transcript = result["results"][0]["alternatives"][0]["transcript"]
            if result["results"][0]['final']:
                self.TARS.user_speaking = False
                self.texts.append(transcript)

    def on_error(self, ws, error):
        print("Error:", error)

    def on_close(self, ws, close_status_code, close_msg):
        print("Closed:", close_status_code, close_msg)

    def on_open(self, ws):
        print("WebSocket connection is open.")
        self.websocket_open = True

        # Send an initial message to configure the Speech to Text service
        config_message = {
            "action": "start",
            "content-type": "audio/l16;rate=44100",
            "continuous": True,
            "interim_results": True,
            "word_confidence": True,
            "timestamps": True,
            "max_alternatives": 3,
            "speaker_labels": True,
            "inactivity_timeout": -1  # No timeout
        }
        ws.send(json.dumps(config_message))

    def audio_stream_callback(self, in_data, frame_count, time_info, status):

        self.audio_data_buffer += in_data
        # print(len(audio_data_buffer))
        # Adjust as needed
        if self.websocket_open and len(self.audio_data_buffer) >= self.CHUNK * 20:
            # Send audio data to the WebSocket when the connection is open
            # save_audio_to_file(audio_data_buffer)
            # audio_data_buffer = b""  # Reset buffer
            self.ws.send(self.audio_data_buffer,
                         opcode=websocket.ABNF.OPCODE_BINARY)
            self.audio_data_buffer = b""  # Reset to bytes

        return in_data, pyaudio.paContinue

    def save_audio_to_file(self, data):
        with open("output_audio.wav", "ab") as audio_file:
            audio_file.write(base64.b64decode(data))

    async def start_recognizing_steam(self):
        self.ws.on_open = self.on_open
        print("URL CONNECTION SUCCESSFUL")
        # Start the WebSocket thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.start()

        # Start the microphone input stream

        print("Microphone stream started. Press Ctrl+C to stop.")
        try:
            while True:
                # if not self.queue_of_texts.empty():
                # await self.get_output()
                await asyncio.sleep(0)
        except KeyboardInterrupt:
            print("Stopping...")

        # Close the microphone stream and WebSocket
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.ws.close()
        ws_thread.join()

    async def get_output(self):
        while True:
            print(self.texts)

class ImageHandler:
    def __init__(self, camera):
        self.camera = camera

    @staticmethod
    def encode_image_to_base64(image: np.ndarray) -> str:
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Could not encode image to JPEG format.")
        return base64.b64encode(buffer).decode('utf-8')

    def capture_screenshot(self) -> str:
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Capture the primary monitor
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)

            # Scale the screenshot to 50% of its original size
            scale_percent = 50  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_screenshot = cv2.resize(
                img, dim, interpolation=cv2.INTER_AREA)

            # Define the file path for the scaled screenshot
            file_path = "captured_screenshot.jpg"
            # Save the scaled screenshot as a JPG file
            cv2.imwrite(file_path, resized_screenshot)
            return file_path

    def capture_camera_image(self) -> str:
        success, image = self.camera.read()
        if success:
            # Scale the image to 50% of its original size
            scale_percent = 50  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_image = cv2.resize(
                image, dim, interpolation=cv2.INTER_AREA)

            # Define the file path for the scaled image
            file_path = "captured_image.jpg"
            # Save the scaled image as a JPG file
            cv2.imwrite(file_path, resized_image)
            return file_path
        else:
            print("Failed to capture image from camera")
            return None


class FaceRecognition:
    def __init__(self, known_faces_file='known_faces.json'):
        self.known_faces_file = known_faces_file
        self.known_faces = self.load_known_faces()

    def load_known_faces(self):
        if os.path.exists(self.known_faces_file):
            with open(self.known_faces_file, 'r') as file:
                data = file.read()
                if not data:  # Checks if the file is empty
                    return {}
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    print("Error decoding JSON from ", self.known_faces_file,
                          ". Initializing with empty data.")
                    return {}
        else:
            # print(f"{self.known_faces_file} not found. Initializing with empty data.")
            return {}

    def save_known_faces(self):
        with open(self.known_faces_file, 'w') as file:
            json.dump(self.known_faces, file)

    def update_known_faces(self, name, face_image):
        if name.lower() == "unknown":
            # print("Skipping update for 'Unknown' face.")
            return

        face_encodings = face_recognition.face_encodings(face_image)
        if face_encodings:
            face_encoding = face_encodings[0]
            self.known_faces[name] = face_encoding.tolist()
            self.save_known_faces()
        else:
            # print(f"No faces detected in the image for {name}.")
            print()

    def label_faces_in_image(self, image):
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(
                list(self.known_faces.values()), face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = list(self.known_faces.keys())[first_match_index]

            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(image, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        return image


class GPT:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if api_key is None:
            raise ValueError("API_KEY is not set")

    def generate_response(self, messages: list) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = self.compose_payload(messages)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()

        if 'choices' in response and response['choices']:
            return response
        else:
            print("Unexpected response format or error:", response)
            return {}

    def compose_payload(self, messages: list) -> dict:
        return {
            "model": "gpt-4-vision-preview", #gpt-3.5-turbo
            "messages": messages,
            "max_tokens": 200,
        }

    @staticmethod
    def extract_token_info(response: dict) -> dict:
        return response.get('usage', {})


class TextToSpeechConverter:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_speech_file(self, text: str, name: str) -> str:
        try:
            print("Generating", name)
            openai.api_key = self.api_key
            speech_file_path = Path(name)

            # Create a speech synthesis response
            response = openai.audio.speech.create(
                model="tts-1-hd",
                voice="nova",
                input=text
            )

            # Save the speech file
            response.stream_to_file(speech_file_path)

            return speech_file_path
        except Exception as e:
            print("Error,", e)

    def play_audio(self, file_path: str) -> None:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Load and play the speech file
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Cleanup
        pygame.mixer.music.stop()
        pygame.mixer.quit()

        # Delete the speech file
        os.remove(file_path)


class ElevenLabsTTSConverter:
    def __init__(self, voice_id: str):
        self.voice_id = voice_id
        self.voice_settings = {
            "stability": 0.08,
            "similarity_boost": 1.0,
            "speed": 2.0,
            "pitch": 1.0,
            "volume": 2.0,
            "emotion": "neutral"
        }

    def generate_speech(self, text: str) -> bytes:
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": self.voice_settings
        }
        url = f'https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}'

        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.content  # Returns the audio content as bytes
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

    def generate_speech_file(self, text: str) -> str:
        audio_content = self.generate_speech(text)
        if audio_content:
            file_path = "elevenlabs_speech.mp3"
            with open(file_path, 'wb') as f:
                f.write(audio_content)
            return file_path
        else:
            print("Failed to generate speech")
            return None

    def play_audio(self, file_path: str) -> None:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Load and play the speech file
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Cleanup
        pygame.mixer.music.stop()
        pygame.mixer.quit()

        # Delete the speech file
        os.remove(file_path)

class PlayHTTTSConverter:
    def __init__(self):
        self.api_key = "702cc6db70e14cbc9fe7719775c83495"
        self.user_id = "Ptxi9NyvbkcOabhm8ev9NLEBOLF2"

    def generate_speech_file(self, text: str, name: str):
        url = "https://api.play.ht/api/v2/tts/stream"
        payload = {
            "text": text,
            "voice": "s3://voice-cloning-zero-shot/60b8791c-f80c-4be2-9db8-5de3cbfbf659/tars/manifest.json",
            "output_format": "mp3",
            "quality": "high",
            "voice_engine": "PlayHT2.0-turbo",
            "emotion": "male_happy", #try male_sad or male_angry etc
            "voice_guidance": 3,
            "style_guidance": 3,
            "text_guidance": 1.5
                }
        headers = {
            "accept": "audio/mpeg",
            "content-type": "application/json",
            "AUTHORIZATION": "702cc6db70e14cbc9fe7719775c83495",
            "X-USER-ID": "Ptxi9NyvbkcOabhm8ev9NLEBOLF2"
        }

        response = requests.post(url, json=payload, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Convert the MP3 byte stream to an AudioSegment
            audio_data = io.BytesIO(response.content)
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")

            # Play the audio segment
            play(audio_segment)
        else:
            print("Failed to get response from API. Status code:", response.status_code)
    def play_audio(self, file_path: str) -> None:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Load and play the speech file
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Cleanup
        pygame.mixer.music.stop()
        pygame.mixer.quit()

        # Delete the speech file
        os.remove(file_path)


class UnifiedTTS:
    def __init__(self, api_key: str, tts_service='openai'):
        self.tts_service = tts_service
        if tts_service == 'openai':
            self.tts_converter = TextToSpeechConverter(api_key)
        elif tts_service == 'elevenlabs':
            self.tts_converter = ElevenLabsTTSConverter(elevenlabs_voice_id)
        elif tts_service == 'playht':
            self.tts_converter = PlayHTTTSConverter()

    def generate_speech_file(self, text: str, name: str, voice_model: str = None) -> str:
        if self.tts_service == 'playht' and voice_model:
            return self.tts_converter.generate_speech_file(text)
        else:
            return self.tts_converter.generate_speech_file(text, name)


class LTM:
    def __init__(self, api_key: str, dimension: int = 1536) -> None:
        self.api_key = api_key
        openai.api_key = self.api_key
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.text_to_index = {}  # Maps text to its index in the Faiss index
        self.index_to_text = []  # Maps index in Faiss to text

    def generate_embedding(self, text: str) -> np.ndarray:
        response = openai.embeddings.create(
            input=text, model="text-embedding-ada-002")
        embedding = np.array(response.data[0].embedding).astype('float32')
        return embedding

    def add_embedding_to_index(self, text: str) -> None:
        if text not in self.text_to_index:
            embedding = self.generate_embedding(text)
            self.index.add(embedding.reshape(1, -1))
            index = len(self.index_to_text)
            self.text_to_index[text] = index
            self.index_to_text.append(text)

    def search_similar_texts(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        if query_text not in self.text_to_index:
            self.add_embedding_to_index(query_text)

        query_embedding = self.generate_embedding(query_text)
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), k)
        # print([(self.index_to_text[idx], distances[0][i]) for i, idx in enumerate(indices[0])])
        return [(self.index_to_text[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

    def store_dialogue_turn(self, user_text: str, ai_text: str) -> None:
        combined_text = user_text + " " + ai_text
        self.add_embedding_to_index(combined_text)

    def save_to_disk(self):
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.mapping_file, 'wb') as f:
                pickle.dump(self.index_to_text, f)
            print("LTM data saved to ", self.index_file,
                  " and ", "{self.mapping_file}.")
        except Exception as e:
            print(f"Error saving LTM data: {e}")

    def load_from_disk(self, index_file, mapping_file):
        if os.path.exists(index_file) and os.path.exists(mapping_file):
            self.index = faiss.read_index(index_file)
            with open(mapping_file, 'rb') as f:
                self.index_to_text = pickle.load(f)
            self.text_to_index = {text: i for i,
                                  text in enumerate(self.index_to_text)}
        else:
            print(f"Index or mapping file not found. Initializing new LTM data.")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.text_to_index = {}
            self.index_to_text = []


class TARS:
    def __init__(self, api_key: str, tts_service='playht', index_file='faiss_index.idx', mapping_file='text_mapping.pkl'):
        # self.speech_recognizer = ContinuousSpeechRecognizer()
        # self.speech_to_text = SpeechToTextConverter(api_key)
        self.user_speaking = False
        self.tars_speaking = False
        self.interrupted = 0
        # Initialize the camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Failed to open the camera.")

        self.image_handler = ImageHandler(self.camera)
        self.face_recognition = FaceRecognition()
        self.gpt = GPT(api_key)
        self.text_to_speech = UnifiedTTS(api_key, tts_service)
        self.messages = self.load_system_message()
        self.ltm = LTM(api_key)
        self.ltm.load_from_disk(index_file, mapping_file)
        self.noise_playing = True

        self.image_counter = 0

    def handle_user_speech_start(self):
        # This method should be called when user speech is detected
        self.tars_speaking = True

    def handle_user_speech_end(self):
        # This method should be called when user speech ends
        self.tars_speaking = False

    def load_system_message(self):
        try:
            with open('system_message.txt', 'r') as file:
                system_message = file.read()
            return [{"role": "system", "content": system_message}]
        except FileNotFoundError:
            return []

    def process_instruction(self, instruction):
        action = instruction.get("action")
        if action == "update_known_faces":
            # Assuming latest image is stored or can be retrieved
            latest_image = self.get_latest_captured_image()
            if latest_image is not None:
                self.handle_face_update(instruction["data"], latest_image)

    def handle_face_update(self, faces_data, image):
        for face in faces_data:
            name = face["name"]
            # Update the face recognition database with the name and the latest image
            self.face_recognition.update_known_faces(name, image)

    def get_latest_captured_image(self):
        # Retrieve the latest captured image file
        camera_image_file = self.image_handler.capture_camera_image()
        if camera_image_file:
            return cv2.imread(camera_image_file)
        return None

    def decode_base64_image(self, base64_string):
        # Decode the base64 string to an OpenCV image
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def update_messages(self, role: str, content: str, image_base64: str = None):
        message_content = [{"type": "text", "text": content}]
        if image_base64:
            message_content.append({"type": "image_url", "image_url": {
                                   "url": f"data:image/jpeg;base64,{image_base64}"}})
        self.messages.append({"role": role, "content": message_content})

        # Retrieve similar messages for the user's message and update GPT context
        if role == "user":
            similar_texts = self.ltm.search_similar_texts(content, k=5)
            similar_texts_converted = [(text, float(score))
                                       for text, score in similar_texts]
            for text, _ in similar_texts_converted:
                # Add similar texts as system messages
                self.messages.append({"role": "system", "content": [
                                     {"type": "text", "text": text}]})

        # Check for a complete dialogue turn
        if len(self.messages) >= 2 and self.messages[-2]["role"] == "user" and self.messages[-1]["role"] == "assistant":
            user_text = self.messages[-2]["content"][0]["text"]
            ai_text = self.messages[-1]["content"][0]["text"]
            self.ltm.store_dialogue_turn(user_text, ai_text)
            # Save after each turn
            self.ltm.save_to_disk('faiss_index.idx', 'text_mapping.pkl')

    async def audio_processing_parallel(self):
        a = asyncIBM(self)
        task1 = asyncio.create_task(a.start_recognizing_steam())

    def camera_parallel(self):
        camera_image_file = self.image_handler.capture_camera_image()
        camera_image = cv2.imread(
            camera_image_file) if camera_image_file else None
        labeled_camera_image = self.face_recognition.label_faces_in_image(
            camera_image) if camera_image is not None else None
        self.camera_image_base64 = self.image_handler.encode_image_to_base64(
            labeled_camera_image) if labeled_camera_image is not None else None

    def screenshot_parallel(self):
        # Capture and process screenshot
        screenshot_file = self.image_handler.capture_screenshot()
        screenshot_image = cv2.imread(
            screenshot_file) if screenshot_file else None
        self.screenshot_base64 = self.image_handler.encode_image_to_base64(
            screenshot_image) if screenshot_image is not None else None

    def play_thinking_audio(self):
        pygame.mixer.init()
        current_mean_pause = 1.0
        max_pause = 10

        while self.mainThreadRunning:
            if self.should_play_thinking_audio and not self.user_speaking and not self.tars_speaking:
                audio_files = [f for f in os.listdir('thinking_audio') if f.endswith('.mp3')]
                if audio_files:
                    selected_audio = random.choice(audio_files)
                    audio_path = os.path.join('thinking_audio', selected_audio)

                    # Load and play the audio
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()

                    # Wait for the music to finish playing
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)  # Sleep to prevent blocking other operations

                    # Randomly adjust the mean pause duration for the next iteration
                    mean_change = random.uniform(-1, 1)
                    current_mean_pause = min(max(current_mean_pause + mean_change, 0), max_pause)

                    # Sample the actual pause duration from a normal distribution with the current mean
                    actual_pause = max(0, min(random.normalvariate(current_mean_pause, 0.5), max_pause))
                    time.sleep(actual_pause)

            else:
                time.sleep(0.1)

    def play_background_noise(self):
        # Load the background noise
        background_noise = AudioSegment.from_file('background_noise.mp3')
        
        # Set initial volume (in dB)
        initial_volume_dB = -20.0  # Example volume
        background_noise = background_noise + initial_volume_dB

        # Loop the sound indefinitely
        background_noise = background_noise * 1000  # Repeat the sound many times

        # Start playback in a separate thread
        playback_thread = threading.Thread(target=play, args=(background_noise,))
        playback_thread.start()

        # Keep the thread running
        while True:
            if not self.noise_playing:
                # This example does not support pause and resume functionality
                # since pydub.playback.play does not provide pause/resume controls
                pass
            time.sleep(1)  # To prevent high CPU usage

    def activate(self):
        self.mainThreadRunning = True
        self.should_play_thinking_audio = False
        thinking_audio_thread = threading.Thread(target=self.play_thinking_audio)
        thinking_audio_thread.start()
        background_noise_thread = threading.Thread(target=self.play_background_noise)
        background_noise_thread.start()
        # Perform a test capture with the camera
        # ret, test_frame = self.camera.read()
        # if not ret:
        #     print("Failed to capture test frame from the camera.")
        #     return None

        self.transcript = [""]
        print("TARS is active.")
        self.user_speaking = False
        audio_processing = asyncio.run(self.audio_processing_parallel())
        print("RAN THE ASYNCIO THREAD")
        # thread = threading.Thread(target=self.audio_processing_parallel, args=())
        # thread.start()
        self.audios = []
        AudioThread = threading.Thread(target=self.play, args=())
        AudioThread.start()
        AudioThread2 = threading.Thread(target=self.stop, args=())
        AudioThread2.start()
        # print("TRYING TO PLAY THE AUDIO THREAD")
        # audio_thread = asyncio.run(self.play())
        last_request = ""
        print("LAST REQUEST:", last_request)
        try:
            while True:
                if self.transcript[-1] != last_request:
                    self.should_play_thinking_audio = True
                    self.camera_parallel()
                    self.screenshot_parallel()
                    print("S E P E R A T E   R E Q U E S T")
                    last_request = self.transcript[-1]
                    time1 = time.time()
                    self.update_messages(
                        "user", self.transcript[-1], self.camera_image_base64)
                    #if self.screenshot_base64:
                        #self.update_messages(
                            #"user", "User's screen: ", self.screenshot_base64)
                    response = self.gpt.generate_response(self.messages)
                    time_end2 = time.time()
                    message_content = ""
                    if 'choices' in response and response['choices']:
                        gpt_response = response['choices'][0]['message']
                        message_content = gpt_response['content']
                        if 'content' in gpt_response:
                            content = gpt_response['content']
                            # Remove markdown formatting characters if present
                            content = content.replace(
                                '```json\n', '').replace('\n```', '').strip()
                            try:
                                # Attempt to parse the JSON content
                                parsed_content = json.loads(content)
                                # Process JSON response
                                if 'response' in parsed_content:
                                    inner_response = parsed_content['response']
                                    if 'message' in inner_response:
                                        message_content = inner_response['message']
                                        print("TARS: " + message_content)
                                        self.update_messages(
                                            "assistant", message_content)

                                    if 'instructions' in inner_response:
                                        for instruction in inner_response['instructions']:
                                            self.process_instruction(
                                                instruction)
                            except json.JSONDecodeError:
                                # Fallback for handling plain text response
                                print("TARS: " + content)
                                print()
                                self.update_messages("assistant", content)
                        else:
                            print("No valid response content received from GPT.")
                    else:
                        print("Unexpected response format or error:", response)
                        print("Actual response received:",
                              response)  # Debug print

                    time_end = time.time()
                    print("GPT LOOSE BOUND:", (time_end-time1))
                    print("GPT TIGHT BOUND:", (time_end2-time1))
                    # Convert response to speech
                    # print("message content: ",message_content)
                    if message_content != "":
                        thread1 = None
                        spl = message_content.split(".")
                        print(spl)
                        i = 0
                        time1 = 0
                        while i < len(spl) and not self.user_speaking:
                            try:
                                if not self.user_speaking:
                                    self.should_play_thinking_audio = False
                                    if i+1 < len(spl):
                                        thread1 = threading.Thread(target=self.generate, args=(
                                            '.'.join(spl[i:i+2]), str(i)+".mp3",))
                                        # thread1 = threading.Thread(target=self.generate,args=(spl[i],str(i)+".mp3",))
                                        thread1.start()
                                        thread1.join()
                                        i += 2
                                    if i < len(spl):
                                        time1 = time.time()-time1
                                        print("Difference:", time1)
                                        time1 = time.time()
                                        thread1 = threading.Thread(
                                            target=self.generate, args=(spl[i], str(i)+".mp3",))
                                        # thread1 = threading.Thread(target=self.generate,args=(spl[i],str(i)+".mp3",))
                                        thread1.start()
                                        thread1.join()
                                        i += 1
                            except Exception as e:
                                print(e)
        except Exception as e:
            print("some error", e)
            self.mainThreadRunning = False

    def generate(self, text, name):
        try:
            # Generate the audio file and get its path
            audio_file_path = self.text_to_speech.generate_speech_file(text, name)
            audio_file_str = str(audio_file_path) if audio_file_path else None

            # Check if the audio file was created successfully
            if audio_file_str and os.path.exists(audio_file_str):
                # Convert MP3 to WAV
                sound = AudioSegment.from_mp3(audio_file_str)
                wav_file_path = audio_file_str.replace(".mp3", ".wav")
                sound.export(wav_file_path, format="wav")

                # Read and play the WAV file
                try:
                    data_read, fs = sf.read(wav_file_path)
                    if not self.user_speaking:
                        self.audios.append([data_read, fs])
                        if len(self.audios) > 0 and not self.user_speaking:
                            self.tars_speaking = True
                            sd.play(self.audios[0][0], self.audios[0][1])
                            self.audios.pop(0)
                            sd.wait()
                except Exception as e:
                    print(f"Error reading audio file {wav_file_path}: {e}")
            else:
                print(f"Failed to generate or find audio file for text: {text}")
        except Exception as e:
            print(f"Error in generating audio: {e}")

    def play(self):
        while True:
            if len(self.audios) > 0 and not self.user_speaking:
                self.tars_speaking = True
                self.stopabble = True
                sd.play(self.audios[0][0], self.audios[0][1])
                self.audios.pop(0)
                sd.wait()
                self.stoppable = False

    def stop(self):
        while True:
            if self.user_speaking:
                self.tars_speaking = False
                self.audios = []
                if self.stoppable:
                    sd.stop()
            if not self.user_speaking:
                self.stoppable = True


if __name__ == "__main__":
    assistant = TARS(api_key)
    try:
        assistant.activate()
    except Exception as e:
        print(e)
