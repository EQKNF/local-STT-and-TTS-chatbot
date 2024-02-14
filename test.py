from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Preload model
model_path = "C:/Users/emilf/Documents/Projects/models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
llmModel = CTransformers(model=model_path, model_type="mistral", gpu_layers=0)

def llmPrompt(lore, transcribedMessage, model):
    # Construct ChatML prompt
    prompt = f'<system>{lore}</system><user>{transcribedMessage}</user>'

    # Define chain with ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_template(prompt)
    chain = prompt_template | model | StrOutputParser()

    # Invoke chain
    output = chain.invoke({})

    print("LLM Response:")
    print(output)

if __name__ == "__main__":
    # Example lore and transcribed message
    lore = "You are Hawa, an helpful AI assistant. You reply with short, to-the-point answers in a friendly tone."
    transcribedMessage = "Hello, what is your name?"

    # Call llmPrompt function
    llmPrompt(lore, transcribedMessage, llmModel)
