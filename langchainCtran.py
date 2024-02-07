from langchain_community.llms import CTransformers

model_path = "C:/Users/emilf/Documents/Projects/models/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = CTransformers(model=model_path, model_type="mistral", gpu_layers=0)

model_promt: str = "Question: What is the capital of Australia?"

response: str = llm(model_promt)

print(response)