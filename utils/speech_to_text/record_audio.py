import sounddevice as sd
import numpy as np
import keyboard

import wave


test = "test_recording.wav"
sample_rate = 48000
is_recording = False

def record_audio(file_path):
            audio_data = []

            print("Press 't' to start recording")
            keyboard.wait("t")
            print("Recording")

            def callback(indata, frames, time, status):
                if status:
                    print(f"Error in audio stream: {status}")
                    return
                if is_recording:
                    audio_data.append(indata.copy())

            with sd.InputStream(callback=callback, channels=2, samplerate=sample_rate, dtype='int32'):
                is_recording = True
                keyboard.wait("t")
                is_recording = False
            print("Stopped recording")

            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(2)
                wf.setsampwidth(4)
                wf.setframerate(sample_rate)
                wf.writeframes(np.concatenate(audio_data, axis=0).tobytes())
                
            audio_data = []

                
if __name__ == "__main__":
    record_audio(test)