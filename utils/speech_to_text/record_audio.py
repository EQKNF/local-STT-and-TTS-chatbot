import sounddevice as sd
import numpy as np

import keyboard
import wave


test = "here.wav"
sample_rate = 48000
is_recording = False

def record_audio(file_path):
            audio_data = []

            print("Press 't' to start recording")
            keyboard.wait("t")
            print("Recording")

            # Define callback function for audio recording.
            def callback(indata, frames, time, status):
                if status:
                    print(f"Error in audio stream: {status}")
                    return
                if is_recording:
                    audio_data.append(indata.copy())

            # Start recording with higher sample rate and bit depth
            with sd.InputStream(callback=callback, channels=2, samplerate=sample_rate, dtype='int32'):
                is_recording = True
                keyboard.wait("t")
                is_recording = False
            print("Stopped recording")

            # Stream audio directly to the temp file
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(2)
                wf.setsampwidth(4)
                wf.setframerate(sample_rate)
                # Convert the list of arrays to a single numpy array, then write the audio data to the temp file
                wf.writeframes(np.concatenate(audio_data, axis=0).tobytes())
                
            # Empty list for next recording
            audio_data = []

                
if __name__ == "__main__":
    record_audio(test)