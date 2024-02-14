from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Preload model
model_path = "C:/Users/emilf/Documents/Projects/models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
llmModel = CTransformers(model=model_path, model_type="mistral", gpu_layers=0)
lore = "You are Hawa, an helpful AI assistant. You reply with short, to-the-point answers in a friendly tone."
message = "Hello, what is your name?"


#del opp i biter

def process_input(identity, input_text, model):
    # Define prompt and chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{identity}"),
        ("user", "{input}")
    ])
    chain = prompt | model | StrOutputParser()

    # Invoke chain
    return chain.invoke({"identity": identity, "input": input_text})

def llmPrompt(transcribedMessage, lore, model):
    output = process_input(lore, transcribedMessage, model)
    output_without_prefix = output.replace("AI:", "").strip()
    print(output_without_prefix)

    print("LLM done")

if __name__ == "__main__":
    llmPrompt(message, lore, llmModel)