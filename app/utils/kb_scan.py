import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .basic_lib import canonicalize_string
from .io import load_json, save_as_pkl
from .kb_cache import get_cache_dir
from .progress_bar import ProgressBar

META_KEYS = {
    "canonical string",
    "literal string",
    "has inverse",
    "content",
    "relation_has instance pairs_instance pairs",
}


def extract_unique_path_simple(data: dict, fields_to_capture: list) -> list:
    results = []
    context = {}

    def recurse(d):
        if not isinstance(d, dict) or not d:
            # Leaf: record a result if all fields are present
            if all(f in context for f in fields_to_capture):
                results.append(context.copy())
            return

        for k, v in d.items():
            if k in fields_to_capture:
                for sub_key, sub_val in v.items():
                    # Assign the captured field
                    if k == "confidence":
                        context[k] = float(sub_key)
                    elif k == "frequency":
                        context[k] = int(sub_key)
                    else:
                        context[k] = sub_key
                    recurse(sub_val)
                    # backtrack (remove assignment)
                    del context[k]
            elif isinstance(v, dict):
                recurse(v)

    recurse(data)
    return results


def index_kb_concepts(
    kb_full_path: str, cache_dir: str, st_container=None
) -> tuple[dict, list, list, dict]:
    kb_full_path = os.path.join(kb_full_path, "kb")

    # index json files with relative paths
    pbar = ProgressBar(
        desc="Indexing concept files", unit="file", st_container=st_container
    )
    concept_file_map = {}
    for root, _, files in os.walk(kb_full_path):
        rel_root = os.path.relpath(root, kb_full_path)
        for file_name in files:
            if file_name.endswith(".json"):
                concept_name = os.path.splitext(file_name)[0]
                rel_path = (
                    file_name if rel_root == "." else os.path.join(rel_root, file_name)
                )
                concept_file_map[concept_name] = rel_path
                pbar.update(1)
    pbar.close()

    def process_concept_file(
        concept_name_file: tuple[str, str],
    ) -> tuple[list, list, dict]:
        concept_name, file_rel_path = concept_name_file
        file_path = os.path.join(kb_full_path, file_rel_path)

        try:
            conceptData = load_json(file_path)

            triples = []
            triples_detail = []
            for relation, targets in conceptData.items():
                if relation in META_KEYS:
                    continue
                relation_canonical = canonicalize_string(relation)
                for tail, triple_data in targets.items():
                    tail_canonical = canonicalize_string(tail)
                    iterData = triple_data.get("iteration", {})
                    freq_dict = triple_data.get("total frequency", {"0": {}})
                    freq = int(next(iter(freq_dict)))
                    conf_dict = triple_data.get("overall confidence", {"0": {}})
                    conf = float(next(iter(conf_dict)))
                    triples.append(
                        {
                            "h": concept_name,
                            "r": relation_canonical,
                            "t": tail_canonical,
                            "total_frequency": freq,
                            "overall_confidence": conf,
                        }
                    )
                    for k, v in iterData.items():
                        fields_to_capture = [
                            "source",
                            "date",
                            "confidence",
                            "frequency",
                        ]
                        res = extract_unique_path_simple(v, fields_to_capture)
                        triples_detail.extend(
                            [
                                {
                                    "h": concept_name,
                                    "r": relation_canonical,
                                    "t": tail_canonical,
                                    "iter": k,
                                    **hrt,
                                }
                                for hrt in res
                            ]
                        )

            return (
                triples,
                triples_detail,
                {},
            )
        except Exception as e:
            print(f"[load] Error processing {concept_name}: {e}")
            print(traceback.print_exc())
            return [], [], {concept_name: e}

    # process files in parallel
    triples_agg = []
    triples_detail_agg = []
    errors = {}

    pbar = ProgressBar(
        total=len(concept_file_map),
        desc="Processing concept files",
        unit="file",
        st_container=st_container,
    )

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_concept_file, item): item
            for item in concept_file_map.items()
        }

        for future in as_completed(futures):
            triples, triples_detail, error = future.result()
            if triples:
                triples_agg.extend(triples)
                triples_detail_agg.extend(triples_detail)
            else:
                errors.update(error)
            pbar.update(1)

    data = {
        "concept_file_map": concept_file_map,
        "triples": triples_agg,
        "triples_detail": triples_detail_agg,
        "errors": errors,
        "saved_at": datetime.now(),
    }
    out_file = f"app_{data["saved_at"].isoformat()}.pkl"
    try:
        save_as_pkl(
            data,
            os.path.join(
                get_cache_dir(os.path.dirname(kb_full_path), cache_dir), out_file
            ),
        )
    except Exception as e:
        print(f"[save] Cannot export cache: ({out_file}) {e}")
    return concept_file_map, triples_agg, triples_detail_agg, errors
