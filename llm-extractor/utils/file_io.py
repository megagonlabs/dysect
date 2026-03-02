def load_text(file_path):
    """Load text content from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()