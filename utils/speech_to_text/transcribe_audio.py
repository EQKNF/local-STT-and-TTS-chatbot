import whisper

# Preload model
model = whisper.load_model("base")

def transcribe_audio(file_path):
    result = model.transcribe(file_path, fp16=False, language="English")
    message = result["text"]
    print(f"From user: {message}")
    return message