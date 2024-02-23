import pyttsx3
engine = pyttsx3.init()

# Voice
voices = engine.getProperty('voices')
engine.setProperty("voice", voices[0].id)
print(f"Voice.id is {voices[0].id}")

# Rate, default = 200
engine.setProperty("rate", 175)

# Volume, min = 0, max = 1
volume = engine.getProperty("volume")
engine.setProperty("volume", 0.4)

engine.say("Hello, yesterday I red a book.")
engine.runAndWait()
engine.stop()


"""
for voice in voices:
    print(voice, voice.id)
    engine.setProperty("voice", voice.id)
    engine.say("Hello, yesterday I read a book.")
    engine.runAndWait()
    engine.stop()

"""