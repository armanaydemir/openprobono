"""Functions for managing and searching vectors and collections in Milvus."""
from __future__ import annotations

import logging
import os
import time
from json import loads
from typing import TYPE_CHECKING

from langfuse.decorators import langfuse_context, observe
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.db import load_vdb, store_vdb
from app.encoders import embed_strs
from app.loaders import (
    partition_html_str,
    partition_uploadfile,
    quickstart_ocr,
    scrape,
    scrape_with_links,
)
from app.models import EncoderParams, MilvusMetadataEnum, OpenAIModelEnum
from app.splitters import chunk_elements_by_title, chunk_str
from app.summarization import summarize_langchain

if TYPE_CHECKING:
    from fastapi import UploadFile
    from pymilvus.orm.iterator import QueryIterator


connection_args = loads(os.environ["Milvus"])
# test connection to db, also needed to use utility functions
connections.connect(uri=connection_args["uri"], token=connection_args["token"])


# used to cache collection params from firebase
COLLECTION_PARAMS = {}
# for storing session data
SESSION_DATA = "SessionData"
# limit for number of results in queries
MAX_K = 16384

# core features

def load_vdb_param(
    collection_name: str,
    param_name: str,
) -> EncoderParams | MilvusMetadataEnum | list:
    """Load a vector database parameter from firebase.

    Parameters
    ----------
    collection_name : str
        The name of the collection using the parameter
    param_name : str
        The name of the desired parameter value

    Returns
    -------
    EncoderParams | MilvusMetadataEnum | list
        EncoderParams if param_name = "encoder"

        MilvusMetadataEnum if param_name = "metadata_format"

        list if param_name = "fields"

    """
    # check if the params are cached
    if collection_name in COLLECTION_PARAMS and \
        param_name in COLLECTION_PARAMS[collection_name]:

        return COLLECTION_PARAMS[collection_name][param_name]
    if collection_name not in COLLECTION_PARAMS:
        COLLECTION_PARAMS[collection_name] = {}
    param_value = load_vdb(collection_name)[param_name]
    # create the parameter object
    match param_name:
        case "encoder":
            COLLECTION_PARAMS[collection_name][param_name] = EncoderParams(
                **param_value,
            )
        case "metadata_format":
            COLLECTION_PARAMS[collection_name][param_name] = MilvusMetadataEnum(
                param_value,
            )
        case "fields":
            COLLECTION_PARAMS[collection_name][param_name] = param_value
        case _:
            raise ValueError(param_name)
    return COLLECTION_PARAMS[collection_name][param_name]


def create_collection(
    name: str,
    encoder: EncoderParams | None = None,
    description: str = "",
    extra_fields: list[FieldSchema] | None = None,
    metadata_format: MilvusMetadataEnum = MilvusMetadataEnum.json,
) -> Collection:
    """Create a collection with a given name and other parameters.

    Parameters
    ----------
    name : str
        The name of the collection to be created
    encoder : EncoderParams, optional
        The embedding model used to create the vectors,
        by default text-embedding-3-small with 768 dimensions
    description : str, optional
        A description for the collection, by default ""
    extra_fields : list[FieldSchema] | None, optional
        A list of fields to add to the collections schema, by default None
    metadata_format : MilvusMetadataEnum, optional
        The format used to store metadata other than text, by default json
        (a single field called `metadata`)

        If `json`, the `metadata` field will be made automatically

        If `no_field`, `extra_fields` should not contain fields

    Returns
    -------
    pymilvus.Collection
        The created collection. Must call load() before query/search.

    Raises
    ------
    ValueError
        If the collection name already exists

    """
    if utility.has_collection(name):
        already_exists = f"collection named {name} already exists"
        raise ValueError(already_exists)
    encoder = encoder if encoder is not None else EncoderParams()
    db_fields = None

    # define schema
    pk_field = FieldSchema(
        name="pk",
        dtype=DataType.INT64,
        is_primary=True,
        description="The primary key",
        auto_id=True,
    )
    text_field = FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        description="The source text",
        max_length=65535,
    )
    embedding_field = FieldSchema(
        name="vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=encoder.dim,
        description="The embedded text",
    )

    # keep track of how the collection stores metadata
    match metadata_format:
        case MilvusMetadataEnum.json:
            extra_fields = [FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="The associated metadata",
            )]
        case MilvusMetadataEnum.no_field:
            if extra_fields:
                msg = "metadata_format = no_field but extra_fields is not empty"
                raise ValueError(msg)
        case MilvusMetadataEnum.field:
            # field format allows for empty extra_fields or a single json field as long
            # as dynamic fields are enabled, otherwise should be no_field or json
            if extra_fields is None:
                extra_fields = []
            else:
                db_fields = [field.name for field in extra_fields]

    schema = CollectionSchema(
        fields=[pk_field, embedding_field, text_field, *extra_fields],
        auto_id=True,
        enable_dynamic_field=True,
        description=description,
    )

    # create collection
    coll = Collection(name=name, schema=schema)
    # create index for vector field
    # AUTOINDEX is only supported through Zilliz, not standalone Milvus
    auto_index = {
        "index_type": "AUTOINDEX",
        "metric_type": "IP",
    }
    coll.create_index("vector", index_params=auto_index, index_name="auto_index")

    # save params in firebase
    store_vdb(name, encoder, metadata_format, db_fields)
    # cache params in dictionary
    COLLECTION_PARAMS[name] = {
        "encoder": encoder,
        "metadata_format": metadata_format,
    }
    if db_fields is not None:
        COLLECTION_PARAMS[name]["fields"] = db_fields
    return coll


@observe()
def query(
    collection_name: str,
    query: str,
    k: int = 4,
    expr: str = "",
    session_id: str = "",
) -> dict:
    """Run a query on a given collection.

    Parameters
    ----------
    collection_name : str
        the collection to query
    query : str
        the query itself
    k : int, optional
        how many chunks to return, by default 4
    expr : str, optional
        a boolean expression to specify conditions for ANN search, by default ""
    session_id : str, optional
        The session id for filtering session data, by default ""

    Returns
    -------
    dict
        With message and result on success, just message on failure

    """
    coll = Collection(collection_name)
    encoder = load_vdb_param(collection_name, "encoder")
    data = embed_strs([query], encoder)
    search_params = {
        "anns_field": "vector",
        "param": {}, # can customize index params assuming you know index type
        "output_fields": ["text", "pk"],
        "data": data,
        "limit": k,
    }
    metadata_format = load_vdb_param(collection_name, "metadata_format")
    match metadata_format:
        case MilvusMetadataEnum.json:
            search_params["output_fields"] += ["metadata"]
        case MilvusMetadataEnum.field:
            search_params["output_fields"] += load_vdb_param(collection_name, "fields")
    if session_id:
        expr += (" and " if expr else "") + f"session_id=='{session_id}'"
    if expr:
        search_params["expr"] = expr
    res = coll.search(**search_params)
    if res:
        # on success, returns a list containing a single inner list containing
        # result objects
        if len(res) == 1:
            # sort hits by ascending distance
            hits = sorted(
                [hit.to_dict() for hit in res[0]],
                key=lambda h: h["distance"],
            )
            # format output for tracing
            langfuse_context.update_current_observation(
                output=[hit["entity"]["pk"] for hit in hits],
            )
            return {"message": "Success", "result": hits}
        return {"message": "Success", "result": res}
    return {"message": "Failure: unable to complete search"}


def fuzzy_keyword_query(keyword_query: str) -> str:
    """Create a fuzzy* version of a keyword query.

    *Replaces uppercase letters and the first letter of every word
    with a wildcard character (_). Mainly for case-insensitivity.

    Parameters
    ----------
    keyword_query : str
        the original keyword query

    Returns
    -------
    str
        A fuzzy version of the keyword query

    """
    min_length_fuzzy = 6
    keywords = keyword_query.split()
    fuzzy_keywords = [
        "".join([
            c if not c.isupper() or len(kw) < min_length_fuzzy else "_"
            for c in kw
        ])
        for kw in keywords
    ]
    fuzzy_keywords = [
        kw if len(kw) < min_length_fuzzy else "_" + kw[1:]
        for kw in fuzzy_keywords
    ]
    fuzzy_keywords_str = " ".join(fuzzy_keywords)
    fuzzy_keywords_str = fuzzy_keywords_str.replace("%", "\\\\%")
    fuzzy_keywords_str = fuzzy_keywords_str.replace('"', '\\"')
    return fuzzy_keywords_str.replace("'", "\\'")


def source_exists(collection_name: str, url: str) -> bool:
    """Check if a url source exists in a collection.

    Parameters
    ----------
    collection_name : str
        name of collection
    url : str
        source url to check for

    Returns
    -------
    bool
        True if the url is found in the collection, False otherwise

    """
    collection = Collection(collection_name)
    q = collection.query(expr=f"metadata['url']=='{url}'")

    return len(q) > 0

@observe(capture_input=False)
def upload_data(
    collection_name: str,
    data: list[dict],
    batch_size: int = 1000,
) -> dict[str, str]:
    """Upload data to a collection with json format.

    Parameters
    ----------
    collection_name : str
        The name of the Milvus collection.
    data : list[dict]
        The data to upload. Format should match collection schema.
        Example: `[{'vector': [], 'text': '', 'metadata': {}}]`
    batch_size : int, optional
        The number of records to be uploaded at a time, by default 1000.

    Returns
    -------
    dict[str, str]
        With a `message`, `insert_count` on success

    """
    collection = Collection(collection_name)
    pks = []
    insert_count = 0
    for i in range(0, len(data), batch_size):
        batch = data[i: i + batch_size]
        current_batch_size = len(batch)
        res = collection.insert(batch)
        pks += res.primary_keys
        insert_count += res.insert_count
        if res.insert_count != current_batch_size:
            # the upload failed, try deleting any partially uploaded data
            bad_deletes = []
            for pk in pks:
                delete_res = collection.delete(expr=f"pk=={pk}")
                if delete_res.delete_count != 1:
                    bad_deletes.append(pk)
            bad_insert = (
                f"Failure: expected {current_batch_size} insertions but got "
                f"{res.insert_count}. "
            )
            if bad_deletes:
                logging.error(
                    "dangling data",
                    extra={"pks": bad_deletes},
                )
                bad_insert += (
                    "We were unable to delete some of your partially uploaded data. "
                    "This has been logged, and your data will eventually be deleted. "
                    "If you would like more information, please email "
                    "contact@openprobono.com."
                )
            else:
                bad_insert += "Any partially uploaded data has been deleted."
            return {"message": bad_insert}
    return {"message": "Success", "insert_count": insert_count}


@observe(capture_output=False)
def get_expr(collection_name: str, expr: str, batch_size: int = 1000) -> dict:
    """Get database entries according to a boolean expression.

    Parameters
    ----------
    collection_name : str
        The name of a pymilvus.Collection
    expr : str
        A boolean expression to filter database entries
    batch_size: int, optional
        The batch size used to fetch entries from Milvus, defaults to 1000

    Returns
    -------
    dict
        Contains `message`, `result` list if successful

    """
    coll = Collection(collection_name)
    collection_format = load_vdb_param(collection_name, "metadata_format")
    output_fields = ["text", "vector"]
    match collection_format:
        case MilvusMetadataEnum.field:
            output_fields += [*load_vdb_param(collection_name, "fields")]
        case MilvusMetadataEnum.json:
            output_fields += ["metadata"]
    q_iter = coll.query_iterator(
        expr=expr,
        output_fields=output_fields,
        batch_size=batch_size,
    )
    hits = []
    res = q_iter.next()
    while len(res) > 0:
        hits += res
        res = q_iter.next()
    q_iter.close()
    langfuse_context.update_current_observation(
        output=f"{len(hits)} hits",
    )
    return {"message": "Success", "result": hits}


@observe()
def delete_expr(collection_name: str, expr: str) -> dict[str, str]:
    """Delete database entries according to a boolean expression.

    Parameters
    ----------
    collection_name : str
        The name of a pymilvus.Collection
    expr : str
        A boolean expression to filter database entries

    Returns
    -------
    dict[str, str]
        Contains `message`, `delete_count` if successful

    """
    coll = Collection(collection_name)
    ids = coll.delete(expr=expr)
    return {"message": "Success", "delete_count": ids.delete_count}


@observe()
def upsert_expr_json(
    collection_name: str,
    expr: str,
    upsert_data: list[dict],
) -> dict[str, str]:
    """Upsert database entries according to a boolean expression.

    Parameters
    ----------
    collection_name : str
        The name of a pymilvus.Collection
    expr : str
        A boolean expression to filter database entries
    upsert_data : list[dict]
        A list of dicts containing the data to be upserted into Milvus.

    Returns
    -------
    dict[str, str]
        Contains `message` and `insert_count` if successful.

    """
    delete_result = delete_expr(collection_name, expr)
    if delete_result["message"] != "Success":
        return delete_result
    delete_count = delete_result["delete_count"]
    insert_result = upload_data(collection_name, upsert_data)
    return {"delete_count": delete_count, **insert_result}


def fields_to_json(fields_entry: dict) -> dict:
    """Convert a Collection entry from fields to json metadata format.

    Parameters
    ----------
    fields_entry : dict
        The entry from a Collection with fields metadata format

    Returns
    -------
    dict
        The entry with fields replaced with a metadata dictionary
        (json metadata format)

    """
    d = {k: v for k, v in fields_entry.items() if k != "entity"}
    d["entity"] = {
        "text": fields_entry["entity"]["text"],
        "metadata": {
            k: v for k, v in fields_entry["entity"].items() if k != "text"
        },
    }
    return d


def query_iterator(
    collection_name: str,
    expr: str,
    output_fields: list[str],
    batch_size: int,
) -> QueryIterator:
    """Get a query iterator for a collection.

    Parameters
    ----------
    collection_name : str
        The name of the collection
    expr : str
        The boolean expression to filter entities in the collection
    output_fields : list[str]
        the fields to return for each entity
    batch_size : int
        the number of entities to process at a time

    Returns
    -------
    QueryIterator
        An iterator over the chosen entities in the collection

    """
    coll = Collection(collection_name)
    return coll.query_iterator(
        expr=expr,
        output_fields=output_fields,
        batch_size=batch_size,
    )

# application level features

@observe(capture_input=False)
def upload_courtlistener(
    collection_name: str,
    opinion: dict,
    max_chunk_size: int = 10000,
    new_after_n_chars: int = 2500,
    chunk_overlap: int = 1000,
) -> dict:
    """Upload a courtlistener opinion to Milvus.

    Parameters
    ----------
    collection_name : str
        name of collection to upload to
    opinion : dict
        The opinion to upload
    max_chunk_size : int, optional
        the max chunk size to be uploaded to milvus, by default 10000
    new_after_n_chars : int, optional
        cuts off new sections after this size (soft max), see `chunk_by_title`
        for details, by default 2500
    chunk_overlap : int, optional
        chunk overlap to be uploaded to milvus, by default 1000

    Returns
    -------
    dict
        With a `message` indicating success or failure and an `insert_count` on success

    """
    langfuse_context.update_current_observation(
        input={"case_name": opinion["case_name"], "id": opinion["id"]},
    )
    if "text" not in opinion or not opinion["text"]:
        return {"message": "Failure: no opinion text found"}

    # chunk
    elements = partition_html_str(opinion["text"])
    texts, _ = chunk_elements_by_title(
        elements,
        max_chunk_size,
        new_after_n_chars,
        chunk_overlap,
    )

    # metadata
    del opinion["text"]
    #delete fields which are empty or over 1000 characters
    maxlen = 1000
    keys_to_remove = [
        key for key in opinion
        if not opinion[key] or (isinstance(opinion[key], str) and len(opinion[key])) > maxlen
    ]
    for key in keys_to_remove:
        del opinion[key]
    # cited opinions take up a lot of tokens and are included in the text
    if "opinions_cited" in opinion:
        del opinion["opinions_cited"]
    # upload
    vectors = embed_strs(texts, load_vdb_param(collection_name, "encoder"))
    data = [{
        "vector": vectors[i],
        "metadata": opinion,
        "text": texts[i],
    } for i in range(len(texts))]
    return upload_data(collection_name, data)


def crawl_upload_site(collection_name: str, description: str, url: str) -> list[str]:
    create_collection(collection_name, description=description)
    urls = [url]
    new_urls, prev_elements = scrape_with_links(url, urls)
    texts, metadatas = chunk_elements_by_title(prev_elements, 3000, 1000, 300)
    ai_summary = summarize_langchain(texts, OpenAIModelEnum.gpt_4o)
    for metadata in metadatas:
        metadata["ai_summary"] = ai_summary
    encoder = load_vdb_param(collection_name, "encoder")
    vectors = embed_strs(texts, encoder)
    data = [{
        "vector": vectors[i],
        "metadata": metadatas[i],
        "text": texts[i],
    } for i in range(len(texts))]
    upload_data(collection_name, data)
    print("new_urls: ", new_urls)
    while len(new_urls) > 0:
        cur_url = new_urls.pop()
        if url == cur_url[:len(url)]:
            urls.append(cur_url)
            cur_elements = scrape(cur_url)
            new_elements = [
                element for element in cur_elements if element not in prev_elements
            ]
            strs, metadatas = chunk_elements_by_title(new_elements, 3000, 1000, 300)
            ai_summary = summarize_langchain(strs, OpenAIModelEnum.gpt_4o)
            for metadata in metadatas:
                metadata["ai_summary"] = ai_summary
            vectors = embed_strs(strs, encoder)
            data = [{
                "vector": vectors[i],
                "metadata": metadatas[i],
                "text": texts[i],
            } for i in range(len(texts))]
            upload_data(collection_name, data)
            prev_elements = cur_elements
    print(urls)
    return urls

@observe(capture_output=False)
def upload_site(collection_name: str, url: str, max_chars=10000, new_after_n_chars=2500, overlap=500) -> dict[str, str]:
    """Scrape, chunk, summarize, and upload a URLs contents to Milvus.

    Parameters
    ----------
    collection_name : str
        Where the chunks will be uploaded.
    url : str
        The site to scrape.

    Returns
    -------
    dict[str, str]
        With a `message` indicating success or failure

    """
    elements = scrape(url)
    if len(elements) == 0:
        return {"message": f"Failure: no elements found at {url}"}
    texts, metadatas = chunk_elements_by_title(elements, max_chars, new_after_n_chars, overlap)
    vectors = embed_strs(texts, load_vdb_param(collection_name, "encoder"))
    ai_summary = summarize_langchain(texts, OpenAIModelEnum.gpt_4o)
    for metadata in metadatas:
        metadata["timestamp"] = str(time.time())
        metadata["url"] = url
        metadata["ai_summary"] = ai_summary
    data = [{
        "vector": vectors[i],
        "metadata": metadatas[i],
        "text": texts[i],
    } for i in range(len(texts))]
    return upload_data(collection_name, data)


def session_upload_ocr(
    file: UploadFile,
    session_id: str,
    summary: str | None = None,
    max_chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> dict[str, str]:
    """OCR scan, chunk, summarize, and upload a file to Milvus.

    Parameters
    ----------
    file : UploadFile
        The file to scan and upload.
    session_id : str
        The session associated with the file.
    summary : str | None, optional
        A user-written summary of the file, by default None.
    max_chunk_size : int, optional
        The max length of a text chunk in chars, by default 1000.
    chunk_overlap : int, optional
        The number of chars to overlap between chunks, by default 150.

    Returns
    -------
    dict[str, str]
        With a `message` indicating success or failure

    """
    reader = quickstart_ocr(file)
    texts = chunk_str(reader, max_chunk_size, chunk_overlap)
    vectors = embed_strs(texts, load_vdb_param(SESSION_DATA, "encoder"))
    ai_summary = summarize_langchain(texts, OpenAIModelEnum.gpt_4o)
    metadata = {
        "session_id": session_id,
        "source": file.filename,
        "ai_summary": ai_summary,
    }
    if summary is not None:
        metadata["user_summary"] = summary
    # upload
    data = [{
        "vector": vectors[i],
        "metadata": metadata,
        "text": texts[i],
    } for i in range(len(texts))]
    return upload_data(SESSION_DATA, data)


def file_upload(
    file: UploadFile,
    session_id: str,
    summary: str | None = None,
) -> dict[str, str]:
    """Perform an unstructured partition, chunk_by_title, and upload a file to Milvus.

    Parameters
    ----------
    file : UploadFile
        The file to upload.
    session_id : str
        The session associated with the file.
    summary: str, optional
        A summary of the file written by the user, by default None.

    Returns
    -------
    dict[str, str]
        With a `message` indicating success or failure

    """
    # extract text
    elements = partition_uploadfile(file)
    if summary is not None:
        for i in range(len(elements)):
            elements[i].metadata["user_summary"] = summary
    # chunk text
    texts, metadatas = chunk_elements_by_title(elements)
    vectors = embed_strs(texts, load_vdb_param(SESSION_DATA, "encoder"))
    # add session id to metadata
    for i in range(len(metadatas)):
        metadatas[i]["session_id"] = session_id
    # upload
    data = [{
        "vector": vectors[i],
        "metadata": metadatas[i],
        "text": texts[i],
    } for i in range(len(texts))]
    return upload_data(SESSION_DATA, data)


def session_source_summaries(
    session_id: str,
    batch_size: int = 1000,
) -> dict[str, dict[str, str]]:
    """Get AI and user-written summaries of any files from a session.

    Parameters
    ----------
    session_id : str
        The session to search for file summaries.
    batch_size : int, optional
        The number of chunks to return from the query iterator at a time,
        by default 1000.

    Returns
    -------
    dict[str, dict[str, str]]
        {"source": {"ai_summary": "str", "user_summary": "str"}}

    """
    coll = Collection(SESSION_DATA)
    q_iter = coll.query_iterator(
        expr=f"session_id=='{session_id}'",
        output_fields=["metadata"],
        batch_size=batch_size,
    )
    source_summaries = {}
    res = q_iter.next()
    while len(res) > 0:
        for item in res:
            metadata = item["metadata"]
            if metadata["source"] not in source_summaries:
                source_summaries[metadata["source"]] = {
                    "ai_summary": metadata["ai_summary"],
                }
                if "user_summary" in metadata:
                    source_summary = source_summaries[metadata["source"]]
                    source_summary["user_summary"] = metadata["user_summary"]
        res = q_iter.next()
    q_iter.close()
    return source_summaries
