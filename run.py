import sounddevice as sd

import numpy as np
import wave
import keyboard
import os

import utils.llm.llm_chain as llm
import utils.speech_to_text.transcribe as stt


def main():
    # Initialize variables
    sample_rate=48000
    audio_data = []
    is_recording = False

    #test
    llm.llm_prompt(llm.message)

    # Define callback function for audio recording. 
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
            user_message = stt.transcribe_audio(file_path)

            print("Processing response, please wait")
            llm.llm_prompt(user_message)

            audio_data = []
            os.remove(file_path)
                
    except KeyboardInterrupt:
        print("\nRecording interrupted. Exiting.")


if __name__ == "__main__": 
    main()
