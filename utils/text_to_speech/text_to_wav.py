from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset


processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

#audio_path = "speech.wav"
#text_input = "Hello there, how are you?"

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def produce_tts(text_input, audio_path):
    inputs = processor(text=text_input, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # Todo: fix sample_rate and play rate
    sf.write(audio_path, speech.numpy(), samplerate=16000)


"""
if __name__ == "__main__":
    produce_tts(text_input, audio_path)
    """