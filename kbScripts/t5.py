import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import defaultdict
from pprint import pprint
scaler = torch.cuda.amp.GradScaler()

class T5Probe:
    def __init__(self, model_name_or_path="t5-large"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path,model_max_length=512)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, )
        self.model.eval()
        self.model.to(self.device)
        self.model.model_max_length = 512
        print("Using device:", self.device)

    def __call__(self, input_text: str, topk: int = 20, max_new_tokens: int = 10):
        formatted_lines, mappings = self.reformat_and_find_mappings([input_text])
        return self.get_top_predictions(formatted_lines, mappings, topk, max_new_tokens)[0]

    def process_input(self, input_content, numresults):
        lines = input_content.split('\n')
        formatted_lines, mappings = self.reformat_and_find_mappings(lines)
        return self.get_top_predictions(formatted_lines, mappings, numresults)

    def reformat_and_find_mappings(self, lines):
        all_formatted_lines = []
        mappings = defaultdict(lambda: defaultdict(list))
        for line_idx, l in enumerate(lines):
            extra_id = 0
            start_indices = [m.start() for m in re.finditer('\[MASK', l)]
            formatted_l = ''
            last_end_idx = 0
            for start_idx in start_indices:
                end_idx = start_idx + l[start_idx:].find(']') + 1
                mask = l[start_idx:end_idx]
                mappings[mask + f' on line {line_idx + 1}'][line_idx].append(extra_id)
                formatted_l += l[last_end_idx:start_idx] + f'<extra_id_{extra_id}>'
                extra_id += 1
                last_end_idx = end_idx
            formatted_l += l[last_end_idx:]
            all_formatted_lines.append(formatted_l)
        return all_formatted_lines, mappings

    def get_top_predictions(self, lines, mappings, numresults, max_new_tokens):
        outputs = []
        all_extractions = []
        num_beams = 128
        for l in lines:
            inputs = self.tokenizer([l], return_tensors='pt')
            inputs.to(self.device)
            with torch.cuda.amp.autocast(dtype = torch.float):
              generated_ids = self.model.generate(inputs.input_ids,
                                                  max_new_tokens=max_new_tokens,
                                                  num_beams=num_beams,
                                                  repetition_penalty=10.0,
                                                  length_penalty=1.0,
                                                  num_return_sequences=numresults,
                                                  early_stopping=True)
            gen_text = self.tokenizer.batch_decode(generated_ids, clean_up_tokenization_spaces=True)
            gen_text = [x.replace('<pad>', '').replace('</s>', '') for x in gen_text]
            gen_text = [re.split(r'<[^<]+>', x)[1:] for x in gen_text]
            gen_text = [[x.strip() for x in y] for y in gen_text]
            all_extractions.append(gen_text)
        # populating the output
        if numresults > num_beams:
            numresults2output = num_beams
        else:
            numresults2output = numresults
        for mask, appearances in mappings.items():
            for line_idx, ext_ids in appearances.items():
                values = []
                for i in range(numresults2output):
                    values.append({'token': all_extractions[line_idx][i][ext_ids[0]]})
                outputs.append({'mask': mask, 'values': values})
        return outputs

if __name__ == '__main__':
  model = T5Probe()
#   result = model('(algebra, is essential for learning, machine learning), (python, is essential for learning, machine learning), ([MASK], is essential for learning, machine learning)', topk=50)
#   result = model('Q: generate a phase describing a person that is hard working look like? A: ', topk=2, max_new_tokens=200)
#   result = model('person seen as a [MASK] is often perceived as intellgent', topk=50, max_new_tokens=30)
#   result = model('person seen as a hero is perceived as a [MASK] person', topk=50, max_new_tokens=30)
#   result = model('person perceived as a [MASK] person, is often seen as a thief', topk=50, max_new_tokens=50)
#   result = model('becoming a software engineer is motivated by the need of [MASK] and ', topk=50, max_new_tokens=50)
#   result = model('a software engineer will [MASK] to the company ', topk=50, max_new_tokens=50)
#   result = model('becoming a thief is the result of being [MASK] and ', topk=50, max_new_tokens=50)
#   result = model('being hard working is caused by the need of [MASK] and', topk=50, max_new_tokens=20)

  result = model('as a teacher that need money will be [MASK] to ', topk=50, max_new_tokens=50)

  pprint(result)
  print("Done")

  
  
