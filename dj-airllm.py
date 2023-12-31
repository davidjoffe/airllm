#from airllm import AirLLMLlama2
import platform

if platform.system() == 'Darwin':  # Darwin is the name returned for macOS
    # This block will execute if the script is running on macOS
    from airllm.airllm_llama_mlx import AirLLMLlamaMlx as AirLLMModel
else:
    # This block will execute for other operating systems
    from airllm import AirLLMLlama2 as AirLLMModel

# Continue with your script using AirLLMModel, which will refer to the correct class based on the OS

MAX_LENGTH=128
# Big 70B model
#model = AirLLMLlama2("garage-bAInd/Platypus2-70B-instruct")
# Somewhat smaller one to test
#model = AirLLMLlama2("mistralai/Mistral-7B-Instruct-v0.2")
model = AirLLMModel("mistralai/Mistral-7B-Instruct-v0.2")

input_text = [
    'Suggest good ideas to possibly help cure aging by crowdsourcing the problem planet-hunters style?'
    ]

#ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`

input_tokens = model.tokenizer(input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    padding=False,#True,
    max_length=MAX_LENGTH)

#   raise AssertionError("Torch not compiled with CUDA enabled") AssertionError: Torch not compiled with CUDA enabled
#generation_output = model.generate(input_tokens['input_ids'].cuda(),
generation_output = model.generate(input_tokens['input_ids'],
     max_new_tokens=2,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)
