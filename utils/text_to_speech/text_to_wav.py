import torch

import os

process_device = torch.device("cpu")
torch.set_num_threads(4)
local_model_file = "models/silero_tts_model.pt"

if not os.path.isfile(local_model_file):
    torch.hub.download_url_to_file("https://models.silero.ai/models/tts/en/v3_en.pt", local_model_file)  

model = torch.package.PackageImporter(local_model_file).load_pickle("tts_models", "model")
model.to(process_device)

sample_rate = 48000
speaker="en_59" 
#best so far: female: 59, 21,0
#male: 77 

text_input = "Hello there! Currently testing text to speech on my computer. She sells sea shells by the sea shore."
audio_file_path = "misc/response.wav"

def produce_tts(text_input, audio_file_path):
    model.save_wav(text=text_input, speaker=speaker, sample_rate=sample_rate, audio_path=audio_file_path)


if __name__ == "__main__":
    produce_tts(text_input, audio_file_path)
