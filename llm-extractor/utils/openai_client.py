import json
import openai
from pathlib import Path
import os

def get_openai_client() -> openai.OpenAI:
    """
    Returns:
        openai.OpenAI: Configured OpenAI client.
    """
    # Env variables have some extra commas/slashes, so we strip them
    api_key=os.getenv("OPENAI_API_KEY").strip().rstrip(',')
    organization=os.getenv("OPENAI_ORG_ID").strip().rstrip(',')
    base_url=os.getenv("OPENAI_BASE_URL").strip().rstrip('/').rstrip(',')
    return openai.OpenAI(
        api_key=api_key,
        organization=organization,
        base_url=base_url
    )

def get_openai_async_client() -> openai.AsyncOpenAI:
    """
    Args:
        cred_path (str): Path to the JSON file containing the keys.
                         Must include OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_BASE_URL.

    Returns:
        openai.AsyncOpenAI: Configured OpenAI client.
    """
    # Env variables have some extra commas/slashes, so we strip them
    api_key=os.getenv("OPENAI_API_KEY").strip().rstrip(',')
    organization=os.getenv("OPENAI_ORG_ID").strip().rstrip(',')
    base_url=os.getenv("OPENAI_BASE_URL").strip().rstrip('/').rstrip(',')
    return openai.AsyncOpenAI(
        api_key=api_key,
        organization=organization,
        base_url=base_url
    )
