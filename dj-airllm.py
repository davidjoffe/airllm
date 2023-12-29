from airllm import AirLLMLlama2

MAX_LENGTH=128
# Big 70B model
#model = AirLLMLlama2("garage-bAInd/Platypus2-70B-instruct")
# Somewhat smaller one to test
model = AirLLMLlama2("mistralai/Mistral-7B-Instruct-v0.2")

input_text = [
    'Suggest good ideas to possibly help cure aging by crowdsourcing the problem planet-hunters style?'
    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH)

generation_output = model.generate(input_tokens['input_ids'].cuda(),
    max_new_tokens=2,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)
