import os
import torch

process_device = torch.device("cpu")
torch.set_num_threads(4)
local_model_file = "models/silero_tts_model.pt"

if not os.path.isfile(local_model_file):
    torch.hub.download_url_to_file("https://models.silero.ai/models/tts/en/v3_en.pt", local_model_file)  

model = torch.package.PackageImporter(local_model_file).load_pickle("tts_models", "model")
model.to(process_device)

sample_rate = 48000
speaker="en_21" 
#best so far: female: 0, 26, 21, 72, 94, 88, 96, 92, 59
#male: 15, 70, 77, 79, 

text_input = "Hello there! Currently testing text to speech on my computer. She sells sea shells by the sea shore."
audio_file_path = "test/response.wav"

def produce_tts(text_input, audio_file_path):
    model.save_wav(text=text_input, speaker=speaker, sample_rate=sample_rate, audio_path=audio_file_path)


if __name__ == "__main__":
    produce_tts(text_input, audio_file_path)
