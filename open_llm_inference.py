# On clusters

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    s = ''
    for l in lines:
        s += l
    return s

model_path = "/scratch/gpfs/GRIFFITHS/lz3156/resources/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

for sce in ['biathlon', 'canoe', 'tugofwar']:

    prompt = 'Answer the queries in the scenario, return only two lists: the first list is the number estimate for each query and the second list is its standard error. For queries on rank, a higher number means more strength (e.g. 100 is way stronger than 1). For queries on which team wins, use 0-100 scale, where a smaller number means the first team more likely wins. \n\nHere is the scenario: \n\n'
    scenario = read_file(f'scenarios/benchmarks/sc-{sce}.txt')

    input = prompt + scenario

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=2048, # The model supports a context window of up to 8,000 tokens
        temperature=0.6,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))