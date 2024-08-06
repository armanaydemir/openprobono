"""Embed text into vectors, and retrieve embedding models for chains."""
from __future__ import annotations

import time

import tiktoken
import voyageai
from openai import APITimeoutError, OpenAI

from app.models import (
    AnthropicModelEnum,
    EncoderParams,
    GoogleModelEnum,
    OpenAIModelEnum,
    VoyageModelEnum,
)

VOYAGE_MODELS = {
    VoyageModelEnum.large_2_instruct,
    VoyageModelEnum.law,
}

MODEL_CONTEXTLENGTH = {
    OpenAIModelEnum.gpt_3_5: 16385,
    OpenAIModelEnum.gpt_3_5_1106: 16385,
    OpenAIModelEnum.gpt_3_5_instruct: 4096,
    OpenAIModelEnum.gpt_4: 8192,
    OpenAIModelEnum.gpt_4_1106: 128000,
    OpenAIModelEnum.gpt_4_turbo: 128000,
    OpenAIModelEnum.gpt_4o: 128000,
    OpenAIModelEnum.gpt_4o_mini: 128000,
    AnthropicModelEnum.claude_3_5_sonnet: 200000,
    AnthropicModelEnum.claude_3_haiku: 200000,
    AnthropicModelEnum.claude_3_opus: 200000,
    AnthropicModelEnum.claude_3_sonnet: 200000,
    GoogleModelEnum.gemini_1_5_flash: 1000000,
}


def token_count(string: str, model: str) -> int:
    """Return the number of tokens in a text string.

    Third-generation embedding models like text-embedding-3-small
    use the cl100k_base encoding.

    Parameters
    ----------
    string : str
        The string whose tokens will be counted
    model : str
        The embedding model that will accept the string

    Returns
    -------
    int
        The number of tokens in the given string for the given model

    """
    if model not in VOYAGE_MODELS:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(string))
    vo = voyageai.Client()
    return vo.count_tokens([string], model)


def embed_strs(text: list[str], encoder: EncoderParams) -> list:
    """Embeds text from a list where each element is within the model max input length.

    Parameters
    ----------
    text : list[str]
        The text entries to embed
    encoder : EncoderParams
        The embedding model parameters

    Returns
    -------
    list
        A list of lists of floats representing the embedded text

    """
    if encoder.name not in VOYAGE_MODELS:
        return embed_strs_openai(text, encoder)
    return embed_strs_voyage(text, encoder)

def embed_strs_openai(text: list[str], encoder: EncoderParams) -> list:
    """Embed a list of strings using an OpenAI client.

    Parameters
    ----------
    text : list[str]
        The list of strings to embed
    encoder : EncoderParams
        The parameters for an OpenAI embedding model

    Returns
    -------
    list
        A list of lists of floats representing embedded strings

    """
    max_tokens = 8191
    max_array_size = 2048
    i = 0
    data = []
    client = OpenAI()
    num_strings = len(text)
    while i < num_strings:
        j = i
        while j < num_strings and j - i < max_array_size:
            tokens = token_count(text[j], encoder.name)
            if tokens > max_tokens:
                msg = f"str at index {j} is {tokens} tokens but the max is {max_tokens}"
                raise ValueError(msg)
            j += 1
        attempt = 1
        num_attempts = 75
        while attempt < num_attempts:
            try:
                args = {"input": text[i:j], "model": encoder.name}
                if encoder.name != OpenAIModelEnum.embed_ada_2:
                    args["dimensions"] = encoder.dim
                response = client.embeddings.create(**args)
                data += [data.embedding for data in response.data]
                i = j
                break
            except APITimeoutError:
                time.sleep(1)
                attempt += 1
    return data


def embed_strs_voyage(text: list[str], encoder: EncoderParams) -> list:
    """Embed a list of strings using a Voyage client.

    Parameters
    ----------
    text : list[str]
        The list of strings to embed
    encoder : EncoderParams
        The parameters for an OpenAI embedding model

    Returns
    -------
    list
        A list of lists of floats representing embedded strings

    """
    max_array_tokens = 120000 # large 2 (not instruct) supports 320K tokens
    max_tokens = 16000
    max_array_size = 128
    i = 0
    data = []
    client = voyageai.Client()
    num_strings = len(text)
    while i < num_strings:
        array_tokens = 0
        j = i
        # determine how long of an array to upload
        while j < num_strings and j - i < max_array_size:
            tokens = token_count(text[j], encoder.name)
            # determine if a string is too many tokens
            if tokens > max_tokens:
                msg = f"str at index {j} is {tokens} tokens but the max is {max_tokens}"
                raise ValueError(msg)
            # determine if the array is too many tokens
            if array_tokens + tokens > max_array_tokens:
                break
            array_tokens += tokens
            j += 1
        attempt = 1
        num_attempts = 75
        while attempt < num_attempts:
            try:
                response = client.embed(
                    text[i:j],
                    model="voyage-large-2-instruct",
                    input_type="document",
                ).embeddings
                data += response
                i = j
                break
            except APITimeoutError:
                time.sleep(1)
                attempt += 1
    return data


def max_token_indices(
    documents: list[str],
    model: str,
) -> list[int]:
    """Get indices of the maximal ranges of documents within the maximum token limit.

    Parameters
    ----------
    documents : list[str]
        The list of documents to calculate ranges for.
    model : str
        The name of the model to determine the max context length.

    Returns
    -------
    list[int]
        The indices marking the document ranges.

    """
    # hard limit on the number of tokens to be summarized, for cost and rate limits
    tokens = 0
    # count tokens to find the number of documents to summarize
    max_tokens = MODEL_CONTEXTLENGTH[model]
    indices = []
    for i, doc in enumerate(documents, start=1):
        tokens += token_count(doc, model)
        if tokens > max_tokens:
            tokens = 0
            indices.append(i)
    indices.append(len(documents))
    return indices
