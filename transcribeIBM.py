import asyncio

import websocket
import json
import base64
import pyaudio
import threading
import time
from ibm_watson import DiscoveryV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import queue

class asyncIBM:

    def __init__(self):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.audio_data_buffer = b""
        self.websocket_open = False
        self.speaking = False
        self.queue_of_texts = queue.Queue()
        self.ws_url = 'wss://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/a060b1a1-8b30-4191-b335-095ddbf0c1ad/v1/recognize'
        self.another_token = 'eyJraWQiOiIyMDIzMTEwNzA4MzYiLCJhbGciOiJSUzI1NiJ9.eyJpYW1faWQiOiJpYW0tU2VydmljZUlkLWU2YmM1ZTQ1LTcyYWYtNDMzOS1hZWYxLTFjMjRjYzRmMjUwZiIsImlkIjoiaWFtLVNlcnZpY2VJZC1lNmJjNWU0NS03MmFmLTQzMzk\
tYWVmMS0xYzI0Y2M0ZjI1MGYiLCJyZWFsbWlkIjoiaWFtIiwianRpIjoiOWViZWEyNDctNzU0ZS00N2QwLWFjY2EtZDRjYmY0MWRlNWY3IiwiaWRlbnRpZmllciI6IlNlcnZpY2VJZC1lNmJjNWU0NS03MmFmLTQzMzktYWVmMS0xYzI0Y2M0ZjI1MGYiLCJuYW1lIjoi\
QXV0by1nZW5lcmF0ZWQgc2VydmljZSBjcmVkZW50aWFscyIsInN1YiI6IlNlcnZpY2VJZC1lNmJjNWU0NS03MmFmLTQzMzktYWVmMS0xYzI0Y2M0ZjI1MGYiLCJ1bmlxdWVfaW5zdGFuY2VfY3JucyI6WyJjcm46djE6Ymx1ZW1peDpwdWJsaWM6c3BlZWNoLXRvLXRle\
HQ6YXUtc3lkOmEvYjE2YzU3ZTMyMWU1NGI1YmI1YmFlZGUzMTc5M2RhMTg6YTA2MGIxYTEtOGIzMC00MTkxLWIzMzUtMDk1ZGRiZjBjMWFkOjoiXSwic3ViX3R5cGUiOiJTZXJ2aWNlSWQiLCJhdXRobiI6eyJzdWIiOiJTZXJ2aWNlSWQtZTZiYzVlNDUtNzJhZi00Mz\
M5LWFlZjEtMWMyNGNjNGYyNTBmIiwiaWFtX2lkIjoiaWFtLVNlcnZpY2VJZC1lNmJjNWU0NS03MmFmLTQzMzktYWVmMS0xYzI0Y2M0ZjI1MGYiLCJzdWJfdHlwZSI6IlNlcnZpY2VJZCIsIm5hbWUiOiJBdXRvLWdlbmVyYXRlZCBzZXJ2aWNlIGNyZWRlbnRpYWxzIn0\
sImFjY291bnQiOnsidmFsaWQiOnRydWUsImJzcyI6ImIxNmM1N2UzMjFlNTRiNWJiNWJhZWRlMzE3OTNkYTE4IiwiZnJvemVuIjp0cnVlfSwiaWF0IjoxNzAxNDA1OTQ0LCJleHAiOjE3MDE0MDk1NDQsImlzcyI6Imh0dHBzOi8vaWFtLmNsb3VkLmlibS5jb20vaWRl\
bnRpdHkiLCJncmFudF90eXBlIjoidXJuOmlibTpwYXJhbXM6b2F1dGg6Z3JhbnQtdHlwZTphcGlrZXkiLCJzY29wZSI6ImlibSBvcGVuaWQiLCJjbGllbnRfaWQiOiJkZWZhdWx0IiwiYWNyIjoxLCJhbXIiOlsicHdkIl19.fslthlnZkcYrNjskPrvjPRdFq-T9VGnj\
mttViQwP3EMqb9emdqmyLiXDMEcVWy3K7vLuViH4YIWG1XJYCHpenDzjMfuC5tUwyfybooHncjD3r53LZWkFvoEyjx_lXPHmMtsF_VFaMcthNASs5msUGhmHNBYw_IYIve5GhW8TTbtO_SsKG5qWwuy7B8P1g97h0Cnec69aKyEJ7TmhXH-gLU0vufFLSNDCisc7TSOE_\
5mXMYGLi_E-ai52S6NAzgbT7sbQIotLsz1hMsmC4jbyIluU0Tvev-zsATQD_sBtwn7HYsi5nrPLEOhnCi7JMPl64DQt6_64ooVGz8S0nbNd9Q'

        self.wsURI = f'{self.ws_url}/v1/recognize?access_token={self.another_token}&model=en-US_BroadbandModel'
        self.p = pyaudio.PyAudio()
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
        self.speaking = True
        result = json.loads(message)
        if "results" in result:
            transcript = result["results"][0]["alternatives"][0]["transcript"]
            self.queue_of_texts.put(transcript)
            #print("Transcript:", transcript)
        if result["results"][0]['final']:
            self.speaking = False
            #print("S T O P P E D   S P E A K I N G . . . . . . . . . . .")


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
            "inactivity_timeout": 10  # No timeout
        }
        ws.send(json.dumps(config_message))

    def audio_stream_callback(self, in_data, frame_count, time_info, status):

        self.audio_data_buffer += in_data
        # print(len(audio_data_buffer))
        if self.websocket_open and len(self.audio_data_buffer) >= self.CHUNK * 20:  # Adjust as needed
            # Send audio data to the WebSocket when the connection is open
            # save_audio_to_file(audio_data_buffer)
            # audio_data_buffer = b""  # Reset buffer
            self.ws.send(self.audio_data_buffer, opcode=websocket.ABNF.OPCODE_BINARY)
            self.audio_data_buffer = b""  # Reset to bytes

        return in_data, pyaudio.paContinue

    def save_audio_to_file(self, data):
        with open("output_audio.wav", "ab") as audio_file:
            audio_file.write(base64.b64decode(data))

    async def start_recognizing_steam(self):
        print("URL CONNECTION SUCCESSFUL")

        self.ws.on_open = self.on_open
        print("URL CONNECTION SUCCESSFUL")
        # Start the WebSocket thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.start()

        # Start the microphone input stream

        print("Microphone stream started. Press Ctrl+C to stop.")
        try:
            while True:
                #if not self.queue_of_texts.empty():
                    #await self.get_output()
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
            print(self.queue_of_texts.get())


async def main():
    a = asyncIBM()
    task1 = asyncio.create_task(a.start_recognizing_steam())
    task2 = asyncio.create_task(a.get_output())
    await task1
    await task2




asyncio.run(main())









# [auth]
# apikey = IHLa42x2EcLjNPLBgHq55SY1o5m0KGCjM8ayYupyXgOq
# region = us-south

# working bhargav version.


# curl -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=CA6nqYYeVitl4mvomK8U704oEp_NhLR4EyUt6tI_udkR" "https://iam.cloud.ibm.com/identity/token"


# Token generation cmd command
# curl -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={CA6nqYYeVitl4mvomK8U704oEp_NhLR4EyUt6tI_udkR}" "https://iam.cloud.ibm.com/identity/token"
# curl -X POST -header "Content-Type: application/x-www-form-urlencoded" -data "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={IHLa42x2EcLjNPLBgHq55SY1o5m0KGCjM8ayYupyXgOq}" "https://iam.cloud.ibm.com/identity/token"


#list [1,[1,2],[1,2,3],[1,2,3,4]]
