import json
import random

MAX_SAMPLES = 4000

seed_ids = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 30, 31, 32, 34, 35, 36, 43, 44, 47, 48, 52, 53, 55, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 78, 81, 84, 85, 88, 89, 91, 92, 108, 120, 123, 127, 131, 134, 136, 147, 149, 164, 179, 182, 223, 287, 363, 891, 919, 925, 1092, 2029, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740])

def sentences_to_text(sents):
    # sents: list of list of tokens
    return " ".join(" ".join(sent) for sent in sents)


def get_entity_info(vertex_set, idx):
    # vertex_set: list of list of mentions
    # idx: entity index (h or t)
    name = f"ENT_{idx}"
    ent_type = "UNKNOWN"
    try:
        mentions = vertex_set[idx]
        if mentions:
            first = mentions[0]
            name = first.get("name", name)
            # try several common keys for entity type
            ent_type = first.get("type") or first.get("entity_type") or first.get("ner") or ent_type
    except Exception:
        pass
    return name, ent_type


def extract_relations(example, rel_map=None):
    vertex_set = example.get("vertexSet", [])
    sents = example.get("sents", [])
    labels = example.get("labels", [])

    relations = []

    # labels is a list of dicts: {"r": ..., "h": ..., "t": ..., "evidence": [...]}
    for lab in labels:
        r = lab.get("r")
        h = lab.get("h")
        t = lab.get("t")
        evidence_ids = lab.get("evidence", [])

        # Use the new helper to get both name and type
        head_name, head_type = get_entity_info(vertex_set, h)
        tail_name, tail_type = get_entity_info(vertex_set, t)

        evidence_text = []
        for sid in evidence_ids:
            if 0 <= sid < len(sents):
                evidence_text.append(" ".join(sents[sid]))

        relation_text = None
        if rel_map and isinstance(rel_map, dict):
            relation_text = rel_map.get(r, r)

        relations.append({
            "head": head_name,
            "head_type": head_type,
            "tail": tail_name,
            "tail_type": tail_type,
            "relation_id": r,
            "relation": relation_text,
            "triple": f'{head_name}\t{head_type}_{relation_text}_{tail_type}\t{tail_name}',
            "evidence_sent_ids": evidence_ids,
            "evidence": evidence_text
        })

    return relations


def transform_example(example, rel_map=None):
    sents = example.get("sents", [])
    return {
        "title": example.get("title", ""),
        "text": sentences_to_text(sents),
        "relations": extract_relations(example, rel_map=rel_map)
    }


def load_data(path):
    # Try: JSON list or single JSON
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        else:
            return [data]
    except Exception:
        # Fallback: JSONL
        data = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
        return data


def main(input_file, output_file, rel_info_path=None, max_samples=MAX_SAMPLES):
    data = load_data(input_file)
    out = []
    out_2 = []

    # Load relation id -> text mapping if provided
    rel_map = None
    if rel_info_path:
        try:
            with open(rel_info_path, "r", encoding="utf-8") as rf:
                rel_map = json.load(rf)
        except Exception:
            rel_map = None


    random.seed(1) # 3 missing -- in notebook as well
    random.shuffle(data)
    dev_count=0
    
    for i, ex in enumerate(data):
        dir_name = 'eval'
        if i in seed_ids:
            dir_name = 'seed'
        
        if dev_count < 500:
            dir_name = 'dev'
            dev_count += 1

        if i >= max_samples:
            break
        item = transform_example(ex, rel_map=rel_map)
        out.append(item)
        item_2 = {"title": item['title'], "text": item['text']}
        item_2['triples']='\n'.join([x['triple'] for x in item['relations']])
        out_2.append(item_2)
        with open(f"data/docred/triples/{dir_name}/doc_{i}.txt", "w", encoding="utf-8") as f:
            f.write(f"{item_2['triples']}\n")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    # with open(output_file.replace(".json", "_triples.txt"), "w", encoding="utf-8") as f:
    #     for item in out_2:
    #         f.write(f"{item['triples']}\n")


if __name__ == "__main__":
    input_path = "data/docred/train_annotated.json"
    prepared_data_path = "data/docred/prepared_train.json"
    rel_info_path = "data/docred/rel_info.json"
    main(input_path, prepared_data_path, rel_info_path=rel_info_path)
