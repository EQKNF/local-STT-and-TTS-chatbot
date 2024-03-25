import torch
try:
    import ujson as json
except ImportError:
    import json
    
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os, sys, time
from functools import wraps

def timeit(func):
    @wraps(func)  # Ensures that the function metadata is preserved
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to execute")
        return result
    return wrapper

# Help the CLI write unicode characters to terminal
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

# Preload model
model_path = "C:/Users/emilf/Documents/models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
if not os.path.isfile(model_path):
    torch.hub.download_url_to_file("https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf?download=true", model_path) 

config = {"max_new_tokens": 256, "repetition_penalty": 1.1, "stop": "<|im_end|>", "temperature": 0.8}
llm_model = CTransformers(model=model_path, model_type="mistral", gpu_layers=0, config=config)

# Prepare model background and introduction
conversation_history_path = "utils/llm/conversations.json"
user = "Emil"
assistant = "Hannah"
lore = f"You are {assistant}, an helpful AI assistant created by Emil. You reply with brief, to-the-point sentences."
introduction_message = f"Hello I'm {user}, please introduce yourself?"


@timeit
def llm_prompt(transcribed_message):
    current_conversation = []
    # Construct ChatML prompt
    chat_history = get_history()
    
    prompt = f"<|im_start|>system\n{lore}\n<|im_end|>\n{chat_history}\n<|im_start|>user\n{transcribed_message}\n<|im_end|>\n<|im_start|>assistant\n"
    
    # Define chain with ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_template(prompt)
    chain = prompt_template | llm_model | StrOutputParser()
    # Invoke chain
    output_llm = chain.invoke({})
    
    # add to conversation history
    try:
        with open(conversation_history_path, "r") as json_file:
            current_conversation = json.load(json_file)
    except FileNotFoundError:
        current_conversation = []
        
    current_conversation["history"].append({'role': 'user', 'content': f"<|im_start|>user\n{transcribed_message}\n<|im_end|>\n"})
    current_conversation["history"].append({'role': 'assistant', 'content': f"<|im_start|>assistant\n{output_llm}\n<|im_end|>\n"})
    
    with open(conversation_history_path, "w") as json_file:
        json.dump(current_conversation, json_file, indent=4)
    
    print(output_llm)
    return output_llm


def get_history():
    prepare_prompt = []
    try:
        with open(conversation_history_path, "r") as json_file:
            data_conversations = json.load(json_file)
    except FileNotFoundError:
        print("File not found")
        
    history = [conversation["content"] for conversation in data_conversations["history"]]
    for content in history:
        prepare_prompt.append(content)
    
    prompt_total_len = sum(len(content) for content in prepare_prompt)
    while prompt_total_len > 500:
        try:
            # print(total_len)
            # print(len(prompt))
            prepare_prompt.pop(0) #0 for oldest, nothing for newest
            prompt_total_len = sum(len(content) for content in prepare_prompt)
        except:
            print("Error: Prompt too long!")
            
    # Create the ready prompt with the accumulated content
    ready_prompt = f"<|im_start|>system\nBelow is the conversation history sorted from oldest to newest conversation.\n<|im_end|>\n\n" + "".join(prepare_prompt) + f"\n<|im_start|>system\nConversation history end.\n<|im_end|>\n\n"
    #print(ready_prompt)
    return ready_prompt


if __name__ == "__main__":
    #get_history()
    llm_prompt(introduction_message)
