""" This is adapted from Seiji's notebook"""
import json
from datetime import datetime
from vllm import LLM, SamplingParams
import os

# model = "meta-llama/Meta-Llama-3-8B-Instruct"
model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#os.system('export CUDA_VISIBLE_DEVICES=7')
#model = "meta-llama/Meta-Llama-3.1-70B"
# quantization=None
quantization="bitsandbytes"
llm = LLM(
        model=model,
        tensor_parallel_size=1, # the number of GPUs you want use
        quantization=quantization,
        download_dir=os.path.expanduser("~/.cache/vllm"),  # Avoid permission issues
        dtype="float16",
        load_format="bitsandbytes",
        max_model_len=52000,
        rope_scaling={"type": "extended", "factor": 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'},
    )
dateNow = str(datetime.now().isoformat()).split('T')[0]
outputFile = 'testingPrompts'+ dateNow +'.txt'
tokenizer = llm.llm_engine.tokenizer.tokenizer
temperature = .0
top_p = 1.0
max_tokens = 1024
sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens
)
# data = list(map(json.loads, open(input_file)))
#prompts = [
    # tokenizer.apply_chat_template([{"role": "user", "content": x[key]}], tokenize=False) for x in data
    #tokenizer.apply_chat_template([{"role": "user", "content": "Who is the best basketball player in the history? Format your response as a python dictionary that can be converted to json"}], tokenize=False),
    #tokenizer.apply_chat_template([{"role": "user", "content": "Who is the best soccer player in the history?"}], tokenize=False),
    #tokenizer.apply_chat_template([{"role": "user", "content": "Who is the best judo player in the history?"}], tokenize=False),
#]

#responses = llm.generate(prompts, sampling_params=sampling_params)
#with open(outputFile, "w") as file:
#    for ins, response in zip(data, responses):
#        ins.update({
#            "output": response.outputs[0].text.strip(),
#            "logprob": response.outputs[0].cumulative_logprob
#        })
#        print(json.dumps(ins), file=file)
#print(responses[0].outputs[0].text.strip())
#print(responses[0].outputs[0].text.strip(), file=outputFile)
#print(responses, file=outputFile)

