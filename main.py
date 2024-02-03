import whisper
import sounddevice as sd
import numpy as np
import wave
import keyboard
import os
from pydub import AudioSegment, effects
from concurrent.futures import ThreadPoolExecutor

def record_audio(sample_rate=48000, duration_seconds=float('inf')):
    # Initialize variables
    audio_data = []
    is_recording = False

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
            processed_file_path = "processed_audio.wav"
            
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

            # Load the recorded audio for post-processing
            recorded_audio = AudioSegment.from_wav(file_path)

            print("Audio recorded and saved to recorded_audio.wav")


            # Parallelize transcription
            with ThreadPoolExecutor() as executor:
                executor.submit(transcribe_audio, file_path)
                audio_data = []

            os.remove(file_path)
                
    except KeyboardInterrupt:
        print("\nRecording interrupted. Exiting.")


def transcribe_audio(file_pathy):
    model = whisper.load_model("base")
    result = model.transcribe(file_pathy, fp16=False, language="English")
    print(result["text"])
    

if __name__ == "__main__": 
    record_audio()
