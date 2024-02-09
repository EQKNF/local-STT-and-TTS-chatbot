from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate

model_path = "C:/Users/emilf/Documents/Projects/models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model = CTransformers(model=model_path, model_type="mistral", gpu_layers=0)

identity = "You are an helpful AI assistant called Hawa. You reply with brief, to-the-point answers with no elaboration."

model_prompt = "What is the capital of Australia?"

chat_template = PromptTemplate.from_template(
    "{system}\n{human}"
)

messages = chat_template.format(system=identity, human=model_prompt)

print(messages)

response = model.invoke(messages)

print(response)