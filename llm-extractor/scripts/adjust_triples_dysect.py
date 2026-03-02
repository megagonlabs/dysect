import os, re, json, csv
import logging
import argparse
from llm_extractor.utils.file_io import load_text

def get_concept(s):
    if type(s)!=str:
        return None, None, None
    parts = s.split('_')
    if len(parts)==3:
        return parts
    return None, None, None


def adjust_triples(input_dir_path, output_dir_path, prefix = 'resume'):
    os.makedirs(output_dir_path, exist_ok=True)
    
    for filename in os.listdir(input_dir_path):
        if filename.startswith(prefix):
            file_path = os.path.join(input_dir_path, filename)
            if os.path.isfile(file_path):
                accountId = ''
                match = re.search(r'(\d+)\..*', filename)
                if match:
                    accountId = match.group(1)
                with open(file_path, 'r') as file:
                    content=file.read()

                replacements = {
                        '```\n': '',
                        '```': '',
                        '"resume_id"': f'resume_{accountId}',
                    }

                for old, new in replacements.items():
                    content = content.replace(old, new)
                
                data = []
                for line in content.split('\n'):
                    row = line.split('\t')
                    row_len = len(row)
                    if row_len in (0,1):
                        continue
                    elif row_len==2:
                        if '_experience_' in row[0] and re.search(r'_experience_\d+\b', row[0]) is None:
                            split = row[0].split('_experience_')
                            row = [f'{split[0]}_experience_0', f'experience_{split[1]}', row[-1]]
                    elif row_len==4:
                        row=[f'{row[0]}_{row[1]}']+row[2:]
                    
                    if len(row)==3:
                        if row[-1].strip() == '':
                            logging.info(f'Empty value in row! {accountId}')
                            continue
                        data.append('\t'.join(row))
                    else:
                        logging.info(f'Not a valid triple row!{accountId}')
                        logging.info(f'{line=}')
                        logging.info(f'{row=}')

                new_content = '\n'.join(data)+'\n'
                with open(output_dir_path+filename, 'w') as file:
                    file.write(new_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir_path", type=str, default="/path/to/data/")
    parser.add_argument("--domain", type=str, default="healthcare")
    parser.add_argument("--spec", type=str, default="base")
    parser.add_argument("--version", type=str, default="test")
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--file_prefix", type=str, default="resume")
    parser.add_argument("--prompt_path", type=str, default="prompts/prompt.json")
    args = parser.parse_args()
    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum logging level
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(f"logs/{os.path.splitext(os.path.basename(__file__))[0]}.log"),
            logging.StreamHandler()
        ]
    )

    domain = args.domain
    spec = args.spec
    version = args.version
    sample_size = args.sample_size

    model = load_text(f'configs/domains/{domain}/model_name.txt')
    logging.info(f'{model=}')

    data_dir_path = args.data_dir_path
    if not data_dir_path.endswith('/'):
        data_dir_path += '/'
    data_dir_path += 'extractions/'
    # input_dir_path = data_dir_path+'base/v4_4o/'
    input_dir_path = data_dir_path+f'{domain}/{spec}/{version}_{model}/'
    output_dir_path = input_dir_path+'adjusted/'
    prefix = 'doc'

    adjust_triples(input_dir_path, output_dir_path, prefix)  # triples is not used in this function
    logging.info(f"Triples adjusted and saved to: {output_dir_path}")
