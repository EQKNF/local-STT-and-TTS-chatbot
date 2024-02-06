from ctransformers import AutoModelForCausalLM
model_dir = "./health-and-wellness-advisor"
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id ="./openhermes-2.5-mistral-7b.Q4_K_M.gguf", model_type="mistral", gpu_layers=0)

print(llm("Hello"))