import tempfile

import utils.llm.llm_chain as llm
import utils.speech_to_text.record_audio as recorder
import utils.speech_to_text.transcribe_audio as transcriber
import utils.text_to_speech.text_to_wav as ttw
import utils.text_to_speech.play_wav as play


def main():
    # First run through to preload models, and prompt introduction
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_intro_file:
        temp_intro_path = temp_intro_file.name

        intro = llm.llm_prompt(llm.introduction_prompt)
        ttw.produce_tts(intro, temp_intro_path)
        play.play_audio(temp_intro_path)

    try:
        while True:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file, \
                tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_response_file:

                temp_audio_path = temp_audio_file.name
                temp_response_path = temp_response_file.name

                recorder.record_audio(temp_audio_path)
                user_message = transcriber.transcribe_audio(temp_audio_path)

                print("Processing response, please wait")
                llm_response = llm.llm_prompt(user_message)

                ttw.produce_tts(llm_response, temp_response_path)
                play.play_audio(temp_response_path)

    except KeyboardInterrupt:
        print("\nRecording interrupted. Exiting.")

if __name__ == "__main__": 
    main()
