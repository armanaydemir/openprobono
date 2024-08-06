"""Search court opinions using CAP and courtlistener data."""
from __future__ import annotations

import logging
import time

from langfuse.decorators import langfuse_context, observe

from app.courtlistener import courtlistener_collection, courtlistener_search
from app.milvusdb import get_expr, query_iterator, upsert_expr_json
from app.models import ChatModelParams, OpinionSearchRequest
from app.summarization import summarize_opinion

logger = logging.getLogger(__name__)

@observe(capture_output=False)
def opinion_search(request: OpinionSearchRequest) -> list[dict]:
    """Search CAP and courtlistener collections for relevant opinions.

    Parameters
    ----------
    request : OpinionSearchRequest
        The opinion search request object

    Returns
    -------
    list[dict]
        A list of dicts containing the results from the search query

    """
    start = time.time()
    # get courtlistener results
    cl_result = courtlistener_search(request)
    cl_hits = cl_result["result"]
    langfuse_context.update_current_observation(
        output=[hit["entity"]["metadata"]["id"] for hit in cl_hits],
    )
    end = time.time()
    logger.info("opinion search time: %f", end - start)
    return cl_hits


@observe()
def add_opinion_summary(opinion_id: int) -> str:
    """Summarize an opinion and update its entries in Milvus.

    Parameters
    ----------
    opinion_id : int
        The opinion_id of chunks in Milvus to summarize

    Returns
    -------
    str
        The opinion summary

    """
    start = time.time()
    res = get_expr(courtlistener_collection, f"metadata['id']=={opinion_id}")
    hits = res["result"]
    hits = sorted(hits, key=lambda x: x["pk"])
    texts = [hit["text"] for hit in hits]
    summary = summarize_opinion(texts, ChatModelParams(model="gpt-4o"))
    for hit in hits:
        hit["metadata"]["ai_summary"] = summary
    # save the summary to Milvus for future searches
    upsert_expr_json(courtlistener_collection, f"metadata['id']=={opinion_id}", hits)
    end = time.time()
    logger.info("opinion summarization time: %f", end - start)
    return summary


@observe()
def count_opinions() -> int:
    """Count the number of unique opinions in Milvus.

    Returns
    -------
    int
        The number of unique opinions in Milvus

    """
    start = time.time()
    q_iter = query_iterator(courtlistener_collection, "", ["opinion_id"], 16384)
    opinions = set()
    res = q_iter.next()
    while len(res) > 0:
        for hit in res:
            if hit["opinion_id"] not in opinions:
                opinions.add(hit["opinion_id"])
        res = q_iter.next()
    q_iter.close()
    end = time.time()
    logger.info("opinion count time: %f", end - start)
    return len(opinions)
