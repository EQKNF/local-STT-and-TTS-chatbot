# V4
import os
import torch

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = "models/silero_tts_model.pt"

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v3_en.pt',
                                   local_file)  

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

example_text = 'Hello there! Currently testing text to speech on my pc. She sells sea shells by the sea shore'
sample_rate = 48000
speaker='en_1'

audio_paths = model.save_wav(text=example_text,
                             speaker=speaker,
                             sample_rate=sample_rate)
