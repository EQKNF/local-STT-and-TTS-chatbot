import whisper
import sounddevice as sd
import numpy as np
import wave
import keyboard
import os
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

from chain import llmPrompt
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def HAWA():
    # Initialize variables
    sample_rate=48000
    audio_data = []
    is_recording = False
    lore = "You are Hawa, an helpful AI assistant. You reply with short, to-the-point answers in a friendly tone."

    # Preload models
    whisper_model = whisper.load_model("base")

    model_path = "C:/Users/emilf/Documents/Projects/models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
    llmModel = CTransformers(model=model_path, model_type="mistral", gpu_layers=0)

    # Define callback function for audio recording
    def callback(indata, frames, time, status):
        if status:
            print(f"Error in audio stream: {status}")
            return
        if is_recording:
            audio_data.append(indata.copy())

    try:
        while True:
            # Create a new file with a unique name (timestamp-based)
            file_path = "recorded_audio.wav"
    
            # Start listening for the 't' key to start/stop recording
            print("Press 't' to start recording")
            keyboard.wait("t")
            print("Recording")

            # Start recording with higher sample rate and bit depth
            with sd.InputStream(callback=callback, channels=2, samplerate=sample_rate, dtype='int32'):
                is_recording = True

                keyboard.wait("t")
                is_recording = False
            print("Stopped recording")

            # Stream audio directly to a file
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(2)  # Specify the number of channels
                wf.setsampwidth(4)  # 4 bytes for int32
                wf.setframerate(sample_rate)

                # Write the audio data
                wf.writeframes(np.concatenate(audio_data, axis=0).tobytes())
            print("Audio recorded and saved to recorded_audio.wav")


            userMessage = transcribe_audio(file_path, whisper_model)
            
            llmPrompt(userMessage, lore, llmModel)

            audio_data = []
            os.remove(file_path)
                
    except KeyboardInterrupt:
        print("\nRecording interrupted. Exiting.")


def transcribe_audio(file_pathy, whisper_model):
    result = whisper_model.transcribe(file_pathy, fp16=False, language="English")
    message = result["text"]
    print(message)
    return message

#def LLM func here. Try open-hermes-2.5 or Orca-2


if __name__ == "__main__": 
    HAWA()
