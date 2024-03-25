import whisper


model = whisper.load_model("base")

def transcribe_audio(file_path):
    result = model.transcribe(file_path, fp16=False, language="English")
    message = result["text"]
    print(f"From user: {message}")
    return message

if __name__ == "__main__":
    transcribe_audio("C:/Users/emilf/Documents/repo/health-and-wellness-advisor/here.wav")