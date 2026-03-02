import json
import os
import pickle


def save_as_pkl(data: dict, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Data saved to {file_path}")


def load_pkl(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
