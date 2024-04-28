"""Load messages and chat with chat models."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anthropic
import openai
import openai.resources
import requests
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from models import (
    EngineEnum,
    HiveChatModel,
    HuggingFaceChatModel,
)
from prompts import HIVE_QA_PROMPT

if TYPE_CHECKING:
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types.beta.tools import ToolsBetaMessage

HIVE_TASK_URL = "https://api.thehive.ai/api/v1/task/sync"
ANTHROPIC_MSG_URL = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 1000

class ChatModelParams:
    """Define a chat model for RAG."""

    def __init__(self: ChatModelParams, engine: EngineEnum, model: str) -> None:
        """Define parameters for a chat model.

        Parameters
        ----------
        engine : str
            The API/Framework on which the model runs
        model : str
            The name of the model

        """
        self.engine = engine
        self.model = model

def messages(history: list[tuple[str | None, str | None]], engine: EngineEnum) -> list:
    match engine:
        case EngineEnum.openai | EngineEnum.anthropic | EngineEnum.huggingface \
            | EngineEnum.hive:
            return messages_dicts(history)
        case EngineEnum.langchain:
            return messages_langchain(history)
    return []

def messages_dicts(
    history: list[tuple[str | None, str | None]],
) -> list[dict]:
    messages = []
    for tup in history:
        if tup[0]:
            messages.append({"role": "user", "content": tup[0]})
        if tup[1]:
            messages.append({"role": "assistant", "content": tup[1]})
    return messages

def messages_langchain(
        history: list[tuple[str | None, str | None]],
) -> list[BaseMessage]:
    messages = []
    for tup in history[1:len(history) - 1]:
        if tup[0]:
            messages.append(HumanMessage(content=tup[0]))
        if tup[1]:
            messages.append(AIMessage(content=tup[1]))
    return messages

def chat(
    messages: list,
    chatmodel: ChatModelParams,
    **kwargs: dict,
):
    match chatmodel.engine:
        case EngineEnum.hive:
            return chat_hive(messages[-1]["content"], chatmodel.model, **kwargs)
        case EngineEnum.openai:
            return chat_openai(messages, chatmodel.model, **kwargs)
        case EngineEnum.anthropic:
            return chat_anthropic(messages, chatmodel.model, **kwargs)
        case EngineEnum.huggingface:
            return chat_huggingface(messages, chatmodel, **kwargs)
        case EngineEnum.langchain:
            msg = "langchain chat function must be implemented manually"
    raise ValueError(msg)

def chat_hive(
    message: str,
    model: str,
    **kwargs: dict,
) -> tuple[str, list[str]]:
    key = "HIVE_7B_API_KEY" if model == HiveChatModel.HIVE_7B else "HIVE_70B_API_KEY"
    system = kwargs.pop("system", HIVE_QA_PROMPT)
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    temperature = kwargs.pop("temperature", 0.0)
    top_p = kwargs.pop("top_p", 0.95)
    headers = {
        "Accept": "application/json",
        "Authorization": f"Token {os.environ[key]}",
        "Content-Type": "application/json",
    }
    data = {
        "text_data": message,
        "options": {
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "system_prompt": system,
            "roles": {
                "user": "user",
                "model": "assistant",
            },
        },
    }
    response = requests.post(HIVE_TASK_URL, headers=headers, json=data, timeout=120)
    response_json = response.json()
    output = response_json["status"][0]["response"]["output"][0]
    message = output["choices"][0]["message"]
    chunks = output["augmentations"]
    return message, chunks

def chat_openai(
    messages: list[dict],
    model: str,
    **kwargs: dict,
):
    client = kwargs.pop("client", openai.OpenAI())
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    temperature = kwargs.pop("temperature", 0.0)
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

def chat_anthropic(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> AnthropicMessage | ToolsBetaMessage:
    client = kwargs.pop("client", anthropic.Anthropic())
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    tools = kwargs.get("tools", [])
    temperature = kwargs.pop("temperature", 0.0)
    endpoint = client.beta.tools.messages if tools else client.messages
    return endpoint.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

def chat_huggingface(
    messages: list,
    chatmodel: ChatModelParams,
    **kwargs: dict,
):
    match chatmodel.model:
        case HuggingFaceChatModel.LLAMA_3_70B | HuggingFaceChatModel.LLAMA_3_8B:
            return chat_llama(messages, chatmodel.model, **kwargs)
        case HuggingFaceChatModel.PHI_3_128K | HuggingFaceChatModel.PHI_3_4K:
            return chat_phi(messages, chatmodel.model, **kwargs)

def chat_llama(
    messages: list[dict],
    model: str,
    **kwargs: dict,
):
    pipe = pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )
    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    do_sample = kwargs.pop("do_sample", False)
    temperature = kwargs.pop("temperature", 0.01)
    top_p = kwargs.pop("top_p", 0.95)
    outputs = pipe(
        prompt,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        return_full_text=False,
    )
    return outputs[0]["generated_text"]

def chat_phi(
    messages: list[dict],
    model: str,
    temperature: float,
    **kwargs: dict,
):
    lm = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipe = pipeline(
        "text-generation",
        model=lm,
        tokenizer=tokenizer,
    )
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    do_sample = kwargs.pop("do_sample", False)
    temperature = kwargs.pop("temperature", 0.01)
    top_p = kwargs.pop("top_p", 0.95)
    outputs = pipe(
        messages,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        return_full_text=False,
    )
    return outputs[0]["generated_text"]
