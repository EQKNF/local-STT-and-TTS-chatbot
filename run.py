import whisper
import sounddevice as sd
from langchain_community.llms import CTransformers

import numpy as np
import wave
import keyboard
import os

import utils.llm_chain as llm


def main():
    # Initialize variables
    sample_rate=48000
    audio_data = []
    is_recording = False

    lore = "You are Hawa, an helpful AI assistant created by Emil. You reply with brief, to-the-point sentences in under 50 words."
    message = "Hello, please introduce yourself?"

    # Preload models
    whisper_model = whisper.load_model("base")

    model_path = "C:/Users/emilf/Documents/Projects/models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
    config = {"max_new_tokens": 256, "repetition_penalty": 1.1, "stop": "<|im_end|>", "temperature": 0.8}
    llmModel = CTransformers(model=model_path, model_type="mistral", gpu_layers=0, config=config)

    #test
    llm.llmPrompt(lore, message, llmModel)

    # Define callback function for audio recording. 
    #Flytt denne ut av hoved logikk?
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

                # Write the audio data to recorded_audio.wav
                wf.writeframes(np.concatenate(audio_data, axis=0).tobytes())

            # Transcribe the recorded messsage
            userMessage = transcribe_audio(file_path, whisper_model)

            print("Processing response, please wait")
            llm.llmPrompt(lore, userMessage, llmModel)

            audio_data = []
            os.remove(file_path)
                
    except KeyboardInterrupt:
        print("\nRecording interrupted. Exiting.")


def transcribe_audio(file_pathy, whisper_model):
    result = whisper_model.transcribe(file_pathy, fp16=False, language="English")
    message = result["text"]
    print(message)
    return message


if __name__ == "__main__": 
    main()
