import whisper

model = whisper.load_model("base") 
result = model.transcribe("health-and-wellness-advisor\heyhey.m4a", fp16=False, language="English")
print(result["text"])
