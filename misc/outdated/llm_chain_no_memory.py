import torch
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os


# Preload model
model_path = "C:/Users/emilf/Documents/models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
if not os.path.isfile(model_path):
    torch.hub.download_url_to_file("https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf?download=true", model_path) 

config = {"max_new_tokens": 256, "repetition_penalty": 1.1, "stop": "<|im_end|>", "temperature": 0.8}
llm_model = CTransformers(model=model_path, model_type="mistral", gpu_layers=0, config=config)

# Prepare model background and introduction
user = "Emil"
lore = "You are Hannah, an helpful AI assistant created by Emil. You reply with brief, to-the-point sentences."
introduction_prompt = f"Hello I'm {user}, please introduce yourself?"


def llm_prompt(transcribed_message):
    # Construct ChatML prompt
    prompt = f"<|im_start|>system\n{lore}\n<|im_end|>\n<|im_start|>user\n{transcribed_message}\n<|im_end|>\n<|im_start|>assistant\n"

    # Define chain with ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_template(prompt)
    
    chain = prompt_template | llm_model | StrOutputParser()

    # Invoke chain
    output = chain.invoke({})

    print(output)
    return output


if __name__ == "__main__":
    llm_prompt(introduction_prompt)
