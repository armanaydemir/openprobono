"""Summarization functions."""
from __future__ import annotations

from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document as LCDocument
from langfuse.decorators import langfuse_context, observe
from unstructured.documents.elements import Element

from app.chat_models import chat_str_fn, get_langchain_chat_model
from app.encoders import max_token_indices
from app.models import ChatModelParams, SummaryMethodEnum
from app.prompts import (
    OPINION_SUMMARY_MAP_PROMPT,
    OPINION_SUMMARY_REDUCE_PROMPT,
    SUMMARY_MAP_PROMPT,
    SUMMARY_PROMPT,
    SUMMARY_REFINE_PROMPT,
)


def summarize_stuffing_msg(documents: list[str]) -> dict[str, str]:
    """Concatenate the documents into a single document.

    Parameters
    ----------
    documents : list[str]
        The documents to concatenate.

    Returns
    -------
    dict : dict[str, str]
        The summary message.

    """
    text = "\n".join(documents)
    prompt = SUMMARY_PROMPT.format(text=text)
    return {"role":"user", "content":prompt}

def summarize_map_reduce_msg(
    documents: list[str],
    chat_model: ChatModelParams,
    **kwargs: dict,
) -> dict[str, str]:
    """Summarize each document (map) and concatenate the summaries (reduce).

    Parameters
    ----------
    documents : list[str]
        The documents to summarize.
    chat_model : ChatModelParams
        The chat model parameters.
    kwargs : dict, optional
        Keyword arguments for the chat function.

    Returns
    -------
    dict[str, str]
        The summary message.

    """
    chat_fn = chat_str_fn(chat_model)
    summaries = []
    for doc in documents:
        prompt = SUMMARY_MAP_PROMPT.format(text=doc)
        doc_msg = {"role":"user", "content":prompt}
        summary = chat_fn([doc_msg], chat_model.model, **kwargs)
        summaries.append(summary)
    concat_summary = "\n".join(summaries)
    prompt = SUMMARY_PROMPT.format(text=concat_summary)
    return {"role": "user", "content": prompt}

def summarize_refine_msg(
    documents: list[str],
    chat_model: ChatModelParams,
    **kwargs: dict,
) -> dict[str, str]:
    """Summarize each document with the previous summary as context and concatenate.

    Parameters
    ----------
    documents : list[str]
        The documents to summarize.
    chat_model : ChatModelParams
        The chat model parameters.
    kwargs : dict, optional
        Keyword arguments for the chat function.

    Returns
    -------
    dict[str, str]
        The summary message.

    """
    chat_fn = chat_str_fn(chat_model)
    summaries = []
    for i, doc in enumerate(documents):
        if i == 0:
            prompt = SUMMARY_REFINE_PROMPT.format(context="", text=doc)
        else:
            prompt = SUMMARY_REFINE_PROMPT.format(
                context=summaries[i - 1],
                text=doc,
            )
        doc_msg = {"role":"user", "content":prompt}
        summary = chat_fn([doc_msg], chat_model.model, **kwargs)
        summaries.append(summary)
    concat_summary = "\n".join(summaries)
    prompt = SUMMARY_PROMPT.format(text=concat_summary)
    return {"role":"user", "content":prompt}

def summarize_opinion(
    documents: list[str],
    chat_model: ChatModelParams | None = None,
    **kwargs: dict,
) -> str:
    """Summarize a judicial opinion with a custom prompt.

    Parameters
    ----------
    documents : list[str]
        A chunked opinion
    chat_model : ChatModelParams | None, optional
        An LLM and engine to use, by default None
    **kwargs : dict
        Keyword arguments for the chat function.

    Returns
    -------
    str
        A summary

    """
    chat_fn = chat_str_fn(chat_model)
    prompt_msg = {"role": "system", "content": OPINION_SUMMARY_MAP_PROMPT}
    # prepare token-maximized document group messages
    max_indices = max_token_indices(documents, chat_model.model)
    msgs = []
    start_idx = 0
    for max_idx in max_indices:
        msgs.append({
            "role": "user",
            "content": "\n\n".join(documents[start_idx:max_idx]),
        })
        start_idx = max_idx
    summaries = []
    # get the document group summaries
    for msg in msgs:
        summary = chat_fn([prompt_msg, msg], chat_model.model, **kwargs)
        summaries.append(summary)
    if len(summaries) == 1:
        # the opinion fit into a single message, return the summary
        return summaries[0]
    # return a combined summary
    prompt_msg["content"] = OPINION_SUMMARY_REDUCE_PROMPT
    partial_summaries_msg = {"role":"user", "content": "\n\n".join(summaries)}
    return chat_fn(
        [prompt_msg, partial_summaries_msg],
        chat_model.model,
        **kwargs,
    )

def get_summary_message(
    documents: list[str | Element],
    method: SummaryMethodEnum,
    chat_model: ChatModelParams,
    **kwargs: dict,
) -> dict:
    """Get a prompt to summarize documents using the specified model and method.

    Parameters
    ----------
    documents : list[str  |  Element]
       The list of documents to summarize.
    method : str
        The summarization method, must be `stuffing`, `map_reduce`, or `refine`.
    chat_model : ChatModelParams
        The engine and model to use for summarization.
    kwargs : dict, optional
        For the LLM. By default, temperature = 0 and max_tokens = 1000.

    Returns
    -------
    dict
        The final prompt to get a summary of all the documents.

    Raises
    ------
    ValueError
        If chat_model.engine is not `openai` or `anthropic`.
    ValueError
        if method is not `stuffing`, `map_reduce`, or `refine`.

    """
    # convert documents to text if elements were given
    if isinstance(documents[0], Element):
        documents = [doc.text for doc in documents]
    # summarize by method
    match method:
        case SummaryMethodEnum.stuffing:
            msg = summarize_stuffing_msg(documents)
        case SummaryMethodEnum.map_reduce:
            msg = summarize_map_reduce_msg(documents, chat_model, **kwargs)
        case SummaryMethodEnum.refine:
            msg = summarize_refine_msg(documents, chat_model, **kwargs)
        case _:
            raise ValueError(method)
    return msg

@observe(capture_input=False)
def summarize(
    documents: list[str | Element | LCDocument],
    method: str = SummaryMethodEnum.stuffing,
    chat_model: ChatModelParams | None = None,
    **kwargs: dict,
) -> str:
    """Summarize text using the specified model and method.

    Parameters
    ----------
    documents : list[str  |  Element  |  LCDocument]
        The list of documents to summarize.
    method : str, optional
        The summarization method, by default `stuffing`.
    chat_model : ChatModelParams, optional
        The engine and model to use for summarization, by default `langchain` and
        `gpt-3.5-turbo-0125`.
    kwargs : dict, optional
        For the LLM. By default, temperature = 0 and max_tokens = 1000.

    Returns
    -------
    str
        The summarized text.

    Raises
    ------
    ValueError
        If chat_model.engine is not `openai`, `anthropic`, or `langchain`.

    """
    chat_model = ChatModelParams() if chat_model is None else chat_model
    chat_fn = chat_str_fn(chat_model)
    msg = get_summary_message(documents, method, chat_model, **kwargs)
    return chat_fn([msg], chat_model.model, **kwargs)

@observe(capture_input=False)
def summarize_langchain(
    documents: list[str],
    model: str,
    **kwargs: dict,
) -> str:
    """Summarize the documents using a chain.

    Parameters
    ----------
    documents : list[str]
        The list of documents to summarize.
    model : str
        The model to use for summarization.
    kwargs : dict, optional
        For the LLM. By default, temperature = 0 and max_tokens = 1000.

    Returns
    -------
    str
        The summarized text.

    """
    max_chunk_indexes = max_token_indices(documents, 120000)
    # convert to Documents if strs were given
    documents = [LCDocument(page_content=doc) for doc in documents]
    chain_type = "stuff"
    chain = load_summarize_chain(
        get_langchain_chat_model(model, **kwargs),
        chain_type=chain_type,
    )
    # get the langchain handler for the current trace
    langfuse_handler = langfuse_context.get_current_langchain_handler()
    # if we need to summarize groups of documents and then summarize those groups
    # because context is too long
    if len(max_chunk_indexes) > 1:
        # summarize each group of documents
        last = 0
        summaries = []
        for max_chunk_index in max_chunk_indexes:
            result = chain.invoke(
                {"input_documents": documents[last: max_chunk_index]},
                config={"callbacks": [langfuse_handler]},
            )
            summaries.append(result["output_text"].strip())
            last = max_chunk_index
        # summarize the document group summaries
        documents = [LCDocument(page_content=doc) for doc in summaries]
    result = chain.invoke(
        {"input_documents": documents},
        config={"callbacks": [langfuse_handler]},
    )
    return result["output_text"].strip()
