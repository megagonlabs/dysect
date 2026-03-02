import os, re, json, time
import openai
from fireworks.client import Fireworks
from llm_extractor.utils.file_io import load_text

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

MAX_TOKENS = 6000
TEMPERATURE = 0.0
SYSTEM_PROMPT = """
You are an information extraction agent designed to improve across iterations.

Your goal is to progressively increase recall while maintaining strict precision.

At each iteration:
- You are given:
  (a) the document text
  (b) previously extracted triples
  (c) optional additional guidance from earlier iterations
- You MUST treat previously extracted triples as already known facts.
- You MUST NOT repeat previously extracted triples.
- You SHOULD look for:
    • relations that were missed earlier
    • entities that were previously unseen
    • new relations involving known entities
    • implicit but explicitly stated facts that can be expressed independently

You must NOT hallucinate, infer unstated facts, or relax schema constraints.
You must obey the allowed concept types, relations, and output format exactly.

Your objective is to extract ONLY new, valid triples that increase coverage of the document.
"""

def setup_gpt():
    with open('~/.cred/openai-key.txt', 'r') as f:
        OPENAI_API_KEY = f.read().strip()
    openai.api_key = OPENAI_API_KEY

def setup_fireworks():
    with open('~/.cred/fireworks-key.txt', 'r') as f:
        fireworks_api_key = f.read().strip()
        fireworks_client = Fireworks(api_key=fireworks_api_key)
    return fireworks_client


def generate(prompt, model, system_prompt=SYSTEM_PROMPT):
    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    response = openai.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages = messages)

    return response.choices[0].message.content

def generate_text(prompt, model, fireworks_client, system_prompt=SYSTEM_PROMPT, max_retries = 5):
    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    for attempt in range(max_retries):
        try:
            response = fireworks_client.chat.completions.create(
                model=model,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                reasoning_effort="none",
                messages = messages)
            out = response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2
                logging.info(f"Error: {e}. Retrying in {wait_time} seconds (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logging.info('Max retries exceeded')
                logging.info('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                logging.info(f"Retires failed for model {model}\nwith prompt {prompt}")
                logging.info('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                return 'Error'

    return out

def read_files_with_prefix(directory, prefix, accountIds=None):
    """
    Reads all files in the given directory that start with the specified prefix.

    Args:
        directory (str): The directory containing the files.
        prefix (str): The prefix to filter files.

    Returns:
        list: A dictionary where keys are filenames and values are file contents.
    """
    data = []

    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                item = {'filename':filename}
                accountId = ''
                # match = re.search(r'^gpt_resume_(\d+)\.txt$', filename)
                match = re.search(r'^doc_(\d+)\.txt$', filename)
                if match:
                      accountId = match.group(1)
                if accountIds and accountId not in accountIds:
                    logging.info(f"Skipping file {filename} as accountId {accountId} is not in the specified accountIds.")
                    continue
                item['accountId']=accountId
                with open(file_path, 'r') as file:
                    item['document']=file.read()
                data.append(item)
                # yield item

    return data

def main(data, prompt, model="gpt-4o-mini", sample_size=0, offset=0, output_dir_path=None, all_extraction_dirs=None, iteration=1, mode='add_triples', system_prompt=""):
    setup_gpt()
    fireworks_client = setup_fireworks()
    prompt_orig = prompt

    extractions = ''
    count = 0

    logging.info(f"****** {iteration=} ******")
    final_dir_path = os.path.join(output_dir_path, mode, str(iteration))
    for sample in data[offset:offset+sample_size]:
        if count >= sample_size:
            logging.info(f"Reached sample size limit of {sample_size}. Stopping further processing.")
            break
        count += 1
        
        logging.info(f"Processing sample {count}: {sample['filename']}")
        
        doc = sample["document"]
        static_example_text='' # Can add any static examples or instructions here if needed
        added_info = ''
        subjects = set()
        rels = set()
        objects = set()
        prev_extractions = []

        for prev_dir in all_extraction_dirs:
            prev_extraction_path = os.path.join(prev_dir, sample['filename'])
            logging.info(f'{prev_extraction_path=}')
            
            with open(prev_extraction_path, 'r') as f:
                try:
                    extractions = json.load(f)
                except Exception:
                    logging.error(f'JSON Load failed on {prev_extraction_path}')
                    extractions = []
                
                for row in extractions:
                    subjects.add(row[0])
                    rels.add(row[2])
                    objects.add(row[3])
                    prev_extractions.append(row)

        # per document
        if mode == 'add_triples':
            if iteration > 0:
                added_info = "### Previously Extracted Triples:\n" + '\n'.join(['\t'.join(row) for row in prev_extractions])
        elif mode == 'add_kb_info':
            # add results of KB based on the document
            import basicLib as bLib

            def getGeneralization(entity, kbID, kbPath):
                return bLib.getValue(entity,'generalizations', kbID, kbPath)

            def getPositiveExamples(entity, kbID, kbPath):
                return bLib.getValue(entity,'specializations', kbID, kbPath)

            def getEntityByConfidence(entity, kbID, kbPath, threshold=0.8):
                res = []
                generalizations = getGeneralization(entity, kbID, kbPath)
                if not generalizations:
                    return res
                logging.info(f'{entity=}')
                for generalization in generalizations:
                    parentGeneralization = getGeneralization(generalization, kbID, kbPath)
                    logging.info(f'{parentGeneralization=}')
                    for parent in parentGeneralization:
                        mutually_exclusive_relation = f'{parent}_is mutually exclusive with_{parent}'
                        negativeExamples = bLib.getValue(generalization, mutually_exclusive_relation, kbID, kbPath)
                        logging.info(f'{negativeExamples=}')
                        if negativeExamples:
                            for negativeExample in negativeExamples:
                                confidence = next(iter(negativeExamples[negativeExample].get('overall confidence', 0)))
                                if float(confidence)>=threshold:
                                    res.append(negativeExample)
                return res
            
            kbID = 'demo_acl_2026_run_dec_25_2025'
            kbPath = 'kbs/'
            threshold = 0.5
            
            samples_to_add = set()
            for subj in subjects | objects:
                logging.info(f'From KB, fetching generalization for concept: {subj}')
                subjectSamples = getGeneralization(subj, kbID, kbPath)
                if not subjectSamples:
                    continue
                for example in subjectSamples:
                    if example == 'Everything':
                        continue
                    confidence = next(iter(subjectSamples[example].get('overall confidence', {'0':0})))
                    if float(confidence) >= threshold:
                        samples_to_add.add(example)
            
            if samples_to_add:
                examplesPrompt = f"### Previously Extracted General Concepts:\n{', '.join(samples_to_add)}\n"
                added_info += examplesPrompt

        prompt = prompt_orig.format(document=doc, example=static_example_text, added_info=added_info)
        logging.info(f"{prompt=}")
        logging.info(f"{added_info=}")
        logging.info(f"{model=}")
        if 'gpt' in model:
            logging.info("Using GPT model")
            out = generate(prompt, model, system_prompt=system_prompt)
        else:
            logging.info("Using Fireworks model")
            out = generate_text(prompt, model, fireworks_client, system_prompt=system_prompt, max_retries=5)
        out = out.strip('```')
        sample['output'] = out
        
        final_dir_path_2 = os.path.join(final_dir_path, 'generalization_concepts')
        output_path = os.path.join(final_dir_path_2, sample['filename'])
        if not os.path.exists(final_dir_path_2):
            logging.info(f"Output directory '{final_dir_path_2}' does not exist. Creating it.")
            os.makedirs(final_dir_path_2, exist_ok=True)
        
        with open(output_path, 'w') as outfile:
            outfile.write(out+'\n')
        logging.info(f"Processed {sample['filename']} and saved output to {output_path}")


if __name__ == "__main__":
    sample_size = 500
    offset=0
    iteration=1
    accountIds=None # set of accountIds to process, or None to process all

    data_dir_path = 'data/docred/'
    input_dir_path = f'{data_dir_path}dev/text/'
    file_prefix = "doc"

    domain = 'general'
    spec = 'base'
    version = 'v1_5_positive'
    # version = 'v1_5_negative'

    mode='add_kb_info'
    kb_mode='generalization_concepts'

    prompt = load_text(f'llm_extractor/configs/domains/docred/prompt_{spec}_{version}.txt')
    logging.info(f'{prompt=}')
    
    # Models = ('gpt-4.1', 'gpt-4.1-mini', 'fireworks/llama-v3p3-70b-instruct', 'fireworks/kimi-k2p5')
    model = load_text('llm_extractor/configs/domains/docred/model_name.txt')
    logging.info(f'{model=}')

    output_dir_path = f'{data_dir_path}dev/extractions/{domain}/{spec}/{model}/{version}'

    all_extraction_dirs = []
    if mode == 'add_kb_info':
        all_extraction_dirs.append(f'data/docred/dev/extractions/general/base/{model}/v1_5_positive/add_triples/0/{kb_mode}')
        ### change base for gpt-4.1
        if iteration > 0:
            all_extraction_dirs.append(f'data/docred/dev/extractions/general/base/{model}/{version}/add_kb_info/{iteration-1}/{kb_mode}')

    logging.info(f'{all_extraction_dirs=}')
    if not os.path.exists(output_dir_path):
        logging.info(f"Output directory '{output_dir_path}' does not exist. Creating it.")
        os.makedirs(output_dir_path, exist_ok=True)

    data = read_files_with_prefix(input_dir_path, file_prefix, accountIds)
    main(data, prompt=prompt, model=model, sample_size=sample_size, offset=offset, output_dir_path=output_dir_path, all_extraction_dirs=all_extraction_dirs, iteration=iteration, mode=mode, system_prompt=SYSTEM_PROMPT)
