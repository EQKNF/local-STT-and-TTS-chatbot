from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager

path = "C:/Users/emilf/TTS/TTS/.models.json"

model_manager = ModelManager(path)

model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")

syn = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path
)

text = "Hello there boy"

output = syn.tts(text)
syn.save_wav(output, "audio.wav")