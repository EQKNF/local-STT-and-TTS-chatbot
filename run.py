import tempfile

import utils.llm.llm_chain as llm
import utils.speech_to_text.record_audio as recorder
import utils.speech_to_text.transcribe_audio as transcriber


def main():
    # First load of llm with lore and introduction request
    llm.llm_prompt(llm.introduction_prompt)
    try:
        while True:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_audio_path = temp_audio_file.name

                recorder.record_audio(temp_audio_path)
                user_message = transcriber.transcribe_audio(temp_audio_path)

                print("Processing response, please wait")
                llm.llm_prompt(user_message)

    except KeyboardInterrupt:
        print("\nRecording interrupted. Exiting.")

if __name__ == "__main__": 
    main()
