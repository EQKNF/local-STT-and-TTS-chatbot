from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Preload model
model_path = "C:/Users/emilf/Documents/Projects/models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
model = CTransformers(model=model_path, model_type="mistral", gpu_layers=0)

def process_input(identity, input_text):
    # Define prompt and chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{identity}"),
        ("user", "{input}")
    ])
    chain = prompt | model | StrOutputParser()

    # Invoke chain
    return chain.invoke({"identity": identity, "input": input_text})

def main():
    lore = "You are Hawa, an helpful AI assistant. You reply with short answers in a friendly tone."
    model_input = "Hello, what is your name?"

    output = process_input(lore, model_input)
    print(output)
    print("Quiting program")

if __name__ == "__main__":
    main()
