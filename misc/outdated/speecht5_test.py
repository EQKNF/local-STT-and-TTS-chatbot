from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
import utils.text_to_speech.play_wav as tts

audio_path = "speech.wav"

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write(audio_path, speech["audio"], samplerate=speech["sampling_rate"])

tts.play_audio(audio_path)