import whisper

# Preload model
whisper_model = whisper.load_model("base")

def transcribe_audio(file_pathy):
    result = whisper_model.transcribe(file_pathy, fp16=False, language="English")
    message = result["text"]
    print(message)
    return message