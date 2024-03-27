import mimetypes
import os
from json import loads
from operator import itemgetter
from typing import List

import encoder
import requests
from bs4 import BeautifulSoup
from fastapi import UploadFile
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from langchain.chains import load_summarize_chain
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.vectorstores import Field, VectorStore, VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.llms import OpenAI as LangChainOpenAI
from langfuse.callback import CallbackHandler
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from unstructured.partition.auto import partition

import prompts

langfuse_handler = CallbackHandler()

connection_args = loads(os.environ["Milvus"])
# test connection to db, also needed to use utility functions
connections.connect(uri=connection_args["uri"], token=connection_args["token"])

project_id = "h2o-gpt"
location = "us"  # Format is "us" or "eu"
processor_id = "c99e554bb49cf45d"


# processor_display_name = "my" # Must be unique per project, e.g.: "My Processor"

def session_upload_str(reader: str, session_id: str, summary: str, max_chunk_size: int = 1000,
                       chunk_overlap: int = 150):
    documents = [
        Document(
            page_content=page,
            metadata={"source": summary, "page": page_number, "session_id": session_id, "user_summary": summary},
        )
        for page_number, page in enumerate([reader], start=1)
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)

    # summarize
    chain = load_summarize_chain(LangChainOpenAI(temperature=0), chain_type="map_reduce")
    result = chain.invoke({"input_documents": documents[:200]}, config={"callbacks": [langfuse_handler]})
    for doc in documents:
        doc.metadata["ai_summary"] = result["output_text"].strip()

    # upload
    ids = load_db(SESSION_PDF).add_documents(documents=documents, embedding=OpenAIEmbeddings(),
                                             connection_args=connection_args)
    num_docs = len(documents)
    if num_docs != len(ids):
        return {
            "message": f"Failure: expected to upload {num_docs} chunk{'s' if num_docs > 1 else ''} for {summary} but "
                       f"got {len(ids)}"}
    return {"message": f"Success: uploaded {summary} as {num_docs} chunk{'s' if num_docs > 1 else ''}"}


def collection_upload_str(reader: str, collection: str, source: str, max_chunk_size: int = 10000,
                          chunk_overlap: int = 1500):
    documents = [
        Document(
            page_content=page,
            metadata={"source": source},
        )
        for page_number, page in enumerate([reader], start=1)
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)

    # summarize
    chain = load_summarize_chain(LangChainOpenAI(temperature=0), chain_type="map_reduce")
    result = chain.invoke({"input_documents": documents[:200]}, config={"callbacks": [langfuse_handler]})
    for doc in documents:
        doc.metadata["ai_summary"] = result["output_text"].strip()

    # upload
    ids = load_db(collection).add_documents(documents=documents, embedding=OpenAIEmbeddings(),
                                            connection_args=connection_args)
    num_docs = len(documents)
    if num_docs != len(ids):
        return {
            "message": f"Failure: expected to upload {num_docs} chunk{'s' if num_docs > 1 else ''} for {source} but got {len(ids)}"}
    return {"message": f"Success: uploaded {source} as {num_docs} chunk{'s' if num_docs > 1 else ''}"}


def scrape(site: str, old_urls: list[str], common_elements: list[str], collection: str, get_links: bool = False):
    print("site: ", site)
    r = requests.get(site)
    site_base = "//".join(site.split("//")[:-1])
    # converting the text 
    s = BeautifulSoup(r.content, "html.parser")
    urls = []

    if get_links:
        for i in s.find_all("a"):
            if "href" in i.attrs:
                href = i.attrs['href']

                if href.startswith("/"):
                    link = site_base + href
                elif href.startswith("http"):
                    link = href
                else:
                    link = old_urls[0]
                    # skip this link

                if link not in old_urls:
                    old_urls.append(link)
                    urls.append(link)

    try:
        elements = partition(url=site)
    except:
        elements = partition(url=site, content_type="text/html")
    e_text = ""
    for el in elements:
        el = str(el)
        if el not in common_elements:
            e_text += el + "\n\n"
    print("elements: ", e_text)
    print("site: ", site)
    collection_upload_str(e_text, collection, site)
    return [urls, elements]


def crawl_and_scrape(site: str, collection: str, description: str):
    create_collection(collection, description)
    urls = [site]
    new_urls, common_elements = scrape(site, urls, [], collection, True)
    print("new_urls: ", new_urls)
    while len(new_urls) > 0:
        cur_url = new_urls.pop()
        if site == cur_url[:len(site)]:
            urls.append(cur_url)
            add_urls, common_elements = scrape(cur_url, urls + new_urls, common_elements, collection)
            new_urls += add_urls
    print(urls)
    return urls


def quickstart_ocr(
        file: UploadFile,
):
    if not file.filename.endswith(".pdf"):
        process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                language_code="en",
                enable_native_pdf_parsing=True,
            )
        )
    else:
        process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                language_code="en",
            )
        )

    # You must set the `api_endpoint`if you use a location other than "us".
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    processor_name = client.processor_path(project_id, location, processor_id)

    # Print the processor information
    print(f"Processor Name: {processor_name}")

    # Load binary data
    raw_document = documentai.RawDocument(
        content=file.file.read(),
        mime_type=mimetypes.guess_type(file.filename)[0],
        # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types
    )

    # Configure the process request
    # `processor.name` is the full resource name of the processor, e.g.:
    # `projects/{project_id}/locations/{location}/processors/{processor_id}`
    request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document, process_options=process_options)

    result = client.process_document(request=request)

    # For a full list of `Document` object attributes, reference this page:
    # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
    document = result.document

    # Read the text recognition output from the processor
    print("The document contains the following text:")
    print(document.text)
    return document.text


# collections by jurisdiction?
US = "USCode"
NC = "NCGeneralStatutes"
CAP = "CAP"
SESSION_PDF = "SessionPDF"
COURTLISTENER = "courtlistener"
COLLECTIONS = {US, NC, CAP, COURTLISTENER}
# collection -> encoder mapping
# TODO: make this a file or use firebase?
COLLECTION_ENCODER = {
    US: encoder.EncoderParams(encoder.OPENAI_3_SMALL, 768),
    NC: encoder.EncoderParams(encoder.OPENAI_3_SMALL, 768),
    CAP: encoder.EncoderParams(encoder.OPENAI_3_SMALL, 768),
    COURTLISTENER: encoder.EncoderParams(encoder.OPENAI_ADA_2, None),
    SESSION_PDF: encoder.EncoderParams(encoder.OPENAI_ADA_2, None)
}

PDF = "PDF"
HTML = "HTML"
COLLECTION_TYPES = {
    US: PDF,
    NC: PDF,
    SESSION_PDF: PDF,
    CAP: CAP,
    COURTLISTENER: COURTLISTENER
}

OUTPUT_FIELDS = {
    PDF: ["source", "page"],
    HTML: [],
    CAP: ["opinion_author", "opinion_type", "case_name_abbreviation", "decision_date", "cite", "court_name",
          "jurisdiction_name"],
    COURTLISTENER: ["source"]
}
# can customize index params with param field assuming you know index type
SEARCH_PARAMS = {
    "anns_field": "vector",
    "param": {},
    "output_fields": ["text"]
}
# AUTOINDEX is only supported through Zilliz, not standalone Milvus
AUTO_INDEX = {
    "index_type": "AUTOINDEX",
    "metric_type": "IP"
}


def create_collection(name: str, description: str = "", extra_fields: list[FieldSchema] = [],
                      params: encoder.EncoderParams = encoder.DEFAULT_PARAMS):
    if utility.has_collection(name):
        print(f"error: collection {name} already exists")
        return

    # TODO: if possible, support custom embedding size for huggingface models
    # TODO: support other OpenAI models
    # define schema, create collection, create index on vectors
    pk_field = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, description="The primary key",
                           auto_id=True)
    # unstructured chunk lengths are sketchy
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, description="The source text", max_length=65535)
    embedding_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=params.dim,
                                  description="The embedded text")
    schema = CollectionSchema(fields=[pk_field, embedding_field, text_field] + extra_fields,
                              auto_id=True, enable_dynamic_field=True, description=description)
    coll = Collection(name=name, schema=schema)
    coll.create_index("vector", index_params=AUTO_INDEX, index_name="auto_index")

    # must call coll.load() before query/search
    return coll


# TODO: custom OpenAIEmbeddings embedding dimensions
def load_db(collection_name: str):
    return Milvus(
        embedding_function=encoder.get_langchain_embedding_function(COLLECTION_ENCODER[collection_name]),
        collection_name=collection_name,
        connection_args=connection_args,
        auto_id=True
    )


def check_params(collection_name: str, query: str, k: int, session_id: str = None):
    if not utility.has_collection(collection_name):
        return {"message": f"Failure: collection {collection_name} not found"}
    if not query or query == "":
        return {"message": "Failure: query not found"}
    if k < 1 or k > 16384:
        return {"message": f"Failure: k = {k} out of range [1, 16384]"}
    if session_id is None and collection_name == SESSION_PDF:
        return {"message": "Failure: session_id not found"}
    if collection_name not in COLLECTION_ENCODER:
        return {"message": f"Failure: encoder for collection {collection_name} not found"}


def query(collection_name: str, q: str, k: int = 4, expr: str = None, session_id: str = None) -> dict:
    """
    This queries the given collection
    Args:
        collection_name: the collection to query
        q: the query itself
        k: how many chunks to return
        expr: a boolean expression to specify conditions for ANN search
        session_id:

    Returns:

    """
    if check_params(collection_name, q, k, session_id):
        return check_params(collection_name, q, k, session_id)

    coll = Collection(collection_name)
    coll.load()
    search_params = SEARCH_PARAMS
    search_params["data"] = encoder.embed_strs([q], COLLECTION_ENCODER[collection_name])
    search_params["limit"] = k
    search_params["output_fields"] += OUTPUT_FIELDS[COLLECTION_TYPES[collection_name]]

    if expr:
        search_params["expr"] = expr
    if session_id:
        session_filter = f"session_id=='{session_id}'"
        # append to existing filter expr or create new filter
        if expr:
            search_params["expr"] += f" and {session_filter}"
        else:
            search_params["expr"] = session_filter
    res = coll.search(**search_params)
    if res:
        # on success, returns a list containing a single inner list containing result objects
        if len(res) == 1:
            hits = res[0]
            return {"message": "Success", "result": hits}
        return {"message": "Success", "result": res}
    return {"message": "Failure: unable to complete search"}


def qa(collection_name: str, query: str, k: int = 4, session_id: str = None):
    """
    Runs query on collection_name and returns an answer along with the top k source chunks

    Args
        collection_name: the name of a pymilvus.Collection
        query: the user query
        k: return the top k chunks
        session_id: the session id for filtering session data

    Returns dict with success message, result, and sources or else failure message
    """
    if check_params(collection_name, query, k, session_id):
        return check_params(collection_name, query, k, session_id)

    db = load_db(collection_name)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tool = llm.bind_tools(
        [prompts.CitedAnswer],
        tool_choice="CitedAnswer",
    )
    output_parser = JsonOutputKeyToolsParser(key_name="CitedAnswer", return_single=True)
    output_fields = OUTPUT_FIELDS[COLLECTION_TYPES[collection_name]]

    def format_docs_with_id(docs: List[Document]) -> str:
        formatted = [
            f"Source ID: {i}\n" + "\n".join([f"Source {field.capitalize()}: {doc.metadata[field]}" for field in
                                             output_fields]) + "\nSource Text: " + doc.page_content
            for i, doc in enumerate(docs)
        ]
        return "\n\n" + "\n\n".join(formatted)

    format_1 = itemgetter("docs") | RunnableLambda(format_docs_with_id)
    answer_1 = prompts.QA_PROMPT | llm_with_tool | output_parser
    if session_id:
        docs = FilteredRetriever(vectorstore=db, session_filter=session_id, search_kwargs={"k": k})
    else:
        docs = db.as_retriever(search_kwargs={"k": k})
    chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=docs)
        .assign(context=format_1)
        .assign(cited_answer=answer_1)
        .pick(["cited_answer", "docs"])
    )
    result = chain.invoke(query)
    cited_sources = [
        {
            field: result["docs"][i].metadata[field]
            for field in output_fields
        }
        for i in result["cited_answer"]["citations"]
    ]
    return {"message": "Success", "result": {"answer": result["cited_answer"]["answer"], "sources": cited_sources}}


def upload_documents(collection_name: str, documents: list[Document]):
    ids = load_db(collection_name).add_documents(documents=documents,
                                                 embedding=encoder.get_langchain_embedding_function(
                                                     COLLECTION_ENCODER[collection_name]),
                                                 connection_args=connection_args)
    num_docs = len(documents)
    if num_docs != len(ids):
        return {"message": f"Failure: expected to upload {num_docs} chunks but got {len(ids)}"}
    return {"message": f"Success: uploaded {num_docs} chunks"}


def delete_expr(collection_name: str, expr: str):
    """
    Deletes database entries according to expr.
    Not atomic, i.e. may only delete some then fail: https://milvus.io/docs/delete_data.md#Delete-Entities.
    
    Args
        collection_name: the name of a pymilvus.Collection
        expr: a boolean expression to specify conditions for ANN search
    """
    if utility.has_collection(collection_name):
        coll = Collection(collection_name)
        coll.load()
        ids = coll.delete(expr=expr)
        return {"message": f"Success: deleted {ids.delete_count} chunks"}


def session_upload_ocr(file: UploadFile, session_id: str, summary: str, max_chunk_size: int = 1000,
                       chunk_overlap: int = 150):
    reader = quickstart_ocr(file)
    documents = [
        Document(
            page_content=page,
            metadata={"source": file.filename, "page": page_number, "session_id": session_id, "user_summary": summary},
        )
        for page_number, page in enumerate([reader], start=1)
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)

    # summarize
    chain = load_summarize_chain(LangChainOpenAI(temperature=0), chain_type="map_reduce")
    result = chain.invoke({"input_documents": documents[:200]})
    for doc in documents:
        doc.metadata["ai_summary"] = result["output_text"].strip()

    # upload
    return upload_documents(SESSION_PDF, documents)


def session_source_summaries(session_id: str, batch_size: int = 1000):
    coll = Collection(SESSION_PDF)
    coll.load()
    q_iter = coll.query_iterator(expr=f"session_id=='{session_id}'",
                                 output_fields=["source", "ai_summary", "user_summary"], batch_size=batch_size)
    source_summaries = {}
    res = q_iter.next()
    while len(res) > 0:
        for item in res:
            if item["source"] not in source_summaries:
                source_summaries[item["source"]] = {"ai_summary": item["ai_summary"]}
                if item["user_summary"] != item["source"]:
                    source_summaries[item["source"]]["user_summary"] = item["user_summary"]
        res = q_iter.next()
    q_iter.close()
    return source_summaries


class FilteredRetriever(VectorStoreRetriever):
    vectorstore: VectorStore
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)
    session_filter: str

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = []
        k = self.search_kwargs["k"]
        # TODO: determine if get_relevant_documents() kwargs param supports filtering by metadata
        # double k on each call to get_relevant_documents() until there are k filtered documents
        while len(docs) < k:
            results = self.vectorstore.as_retriever(search_kwargs={"k": k}).get_relevant_documents(query=query)
            docs += [doc for doc in results if doc.metadata['session_id'] == self.session_filter and doc not in docs]
            k = 2 * k
        return docs[:self.search_kwargs["k"]]