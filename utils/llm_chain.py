from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Preload model
model_path = "C:/Users/emilf/Documents/Projects/models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
config = {"max_new_tokens": 256, "repetition_penalty": 1.1, "stop": "<|im_end|>"}
llmModel = CTransformers(model=model_path, model_type="mistral", gpu_layers=0, config=config)

lore = "You are Hawa, an helpful AI assistant. You reply with brief, to-the-point answers with no elaboration."
message = "Hello, what the capital of Australia and Norway?"

def llmPrompt(lore, transcribedMessage, model):
    # Construct ChatML prompt
    prompt = f"<|im_start|>system\n{lore}\n<|im_end|>\n<|im_start|>user\n{transcribedMessage}\n<|im_end|>\n<|im_start|>assistant\n"

    # Define chain with ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_template(prompt)
    
    chain = prompt_template | model | StrOutputParser()

    # Invoke chain
    output = chain.invoke({})

    print(output)
    return output

if __name__ == "__main__":
    llmPrompt(lore, message, llmModel)
