import whisper
import sounddevice as sd
import numpy as np
import wave
import keyboard
import os
from pydub import AudioSegment, effects
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

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
        while not keyboard.is_pressed("esc"):
            # Create a new file with a unique name (timestamp-based)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"recorded_audio_{timestamp}.wav"
            processed_file_path = f"processed_audio_{timestamp}.wav"
            
            # Start listening for the 't' key to start/stop recording
            print("Press 't' to start/stop recording...")
            keyboard.wait("t")
            print("Started recording")

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

            # Apply post-processing (adjust as needed)
            processed_audio = recorded_audio.set_frame_rate(44100)  # Adjust frame rate if needed

            # Normalize volume (adjust as needed)
            processed_audio = processed_audio.normalize()

            # Apply band-pass filter for basic noise reduction (adjust as needed)
            processed_audio = effects.low_pass_filter(processed_audio, 800)
            processed_audio = effects.high_pass_filter(processed_audio, 100)

            # Export the processed audio to a new file
            processed_audio.export(processed_file_path, format="wav")

            print(f"Audio recorded and saved to {file_path}")
            print(f"Processed audio saved to {processed_file_path}")

            with ThreadPoolExecutor() as executor:
                    executor.submit(transcribe_audio, processed_file_path)

            os.remove(file_path)
            os.remove(processed_file_path)

    except KeyboardInterrupt:
        print("\nRecording interrupted. Exiting.")

def transcribe_audio(file_pathy):
    model = whisper.load_model("base")
    result = model.transcribe(file_pathy, fp16=False, language="English")
    print(result["text"])


if __name__ == "__main__": 
    record_audio()
