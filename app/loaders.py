"""Functions for loading text from files/urls."""
from __future__ import annotations

import ast
import io
import mimetypes
import os
import pathlib
import time
import uuid
from typing import TYPE_CHECKING

import requests
from bs4 import BeautifulSoup
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from langfuse.decorators import observe
from pymilvus import Collection
from pypandoc import ensure_pandoc_installed
from unstructured.partition.auto import partition
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.rtf import partition_rtf

from app.db import store_vdb_chunk, store_vdb_source

if TYPE_CHECKING:
    from fastapi import UploadFile
    from unstructured.documents.elements import Element


def partition_html_str(html: str) -> list[Element]:
    """Partition an HTML string into elements.

    Parameters
    ----------
    html : str
        The HTML string.

    Returns
    -------
    list[Element]
        The extracted elements.

    """
    return partition_html(text=html)


def partition_uploadfile(file: UploadFile) -> list[Element]:
    """Partition an uploaded file into elements.

    Parameters
    ----------
    file : UploadFile
        The file to partition.

    Returns
    -------
    list[Element]
        The extracted elements.

    """
    return partition(file=file.file, metadata_filename=file.filename)


@observe(capture_output=False)
def scrape(site: str) -> list[Element]:
    """Scrape a site for text and partition it into elements.

    Parameters
    ----------
    site : str
        The URL to scrape.

    Returns
    -------
    list[Element]
        The scraped elements.

    """
    try:
        if site.endswith(".pdf"):
            r = requests.get(site, timeout=10)
            elements = partition_pdf(file=io.BytesIO(r.content))
        elif site.endswith(".rtf"):
            r = requests.get(site, timeout=10)
            ensure_pandoc_installed()
            elements = partition_rtf(file=io.BytesIO(r.content))
        else:
            elements = partition(url=site)
    except Exception as error:
        print("Error in regular partition: " + str(error))
        elements = partition(url=site, content_type="text/html")
    return elements


def scrape_with_links(
    site: str,
    old_urls: list[str],
) -> tuple[list[str], list[Element]]:
    """Scrape a site and get any links referenced on the site.

    Parameters
    ----------
    site : str
        The URL to scrape.
    old_urls : list[str]
        The list of URLs already visited.

    Returns
    -------
    tuple[list[str], list[Element]]
        URLs, elements

    """
    print("site: ", site)
    r = requests.get(site, timeout=10)
    site_base = "/".join(site.split("/")[:-1])
    # converting the text
    s = BeautifulSoup(r.content, "html.parser")
    urls = []

    # get links
    for i in s.find_all("a"):
        if "href" in i.attrs:
            href = i.attrs["href"]

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

    elements = scrape(site)
    return urls, elements


def quickstart_ocr(file: UploadFile) -> str:
    """Extract text from a file using OCR.

    Parameters
    ----------
    file : UploadFile
        The file to extract text from.

    Returns
    -------
    str
        The extracted text from the file.

    """
    project_id = "h2o-gpt"
    location = "us"  # Format is "us" or "eu"
    processor_id = "c99e554bb49cf45d"
    if not file.filename.endswith(".pdf"):
        process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                language_code="en",
                enable_native_pdf_parsing=True,
            ),
        )
    else:
        process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                language_code="en",
            ),
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
        # Refer to https://cloud.google.com/document-ai/docs/file-types
        # for supported file types
    )

    # Configure the process request
    # `processor.name` is the full resource name of the processor, e.g.:
    # `projects/{project_id}/locations/{location}/processors/{processor_id}`
    request = documentai.ProcessRequest(
        name=processor_name, raw_document=raw_document,
        process_options=process_options,
    )

    result = client.process_document(request=request)

    # For a full list of `Document` object attributes, reference this page:
    # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
    document = result.document

    # Read the text recognition output from the processor
    print("The document contains the following text:")
    print(document.text)
    return document.text


def transfer_hive(collection_name: str) -> None:
    """Transfer a collection from Milvus to Hive.

    Parameters
    ----------
    collection_name : str
        The name of the collection to transfer.

    """
    headers = {
        "Accept": "application/json",
        "Authorization": f"Token {os.environ['HIVE_ADD_PROJECT_KEY']}",
        "Content-Type": "application/json",
    }
    coll = Collection(collection_name)
    coll.load()
    q_iter = coll.query_iterator(output_fields=["text"])
    res = q_iter.next()
    num_batches = 0
    error = False
    while len(res) > 0:
        print(f"processing batch {num_batches}")
        for i, item in enumerate(res):
            if i % 100 == 0:
                print(f" i = {i}")
            data = {"text_data": item["text"]}
            attempt = 1
            num_attempts = 75
            while attempt < num_attempts:
                try:
                    response = requests.post(
                        "https://api.thehive.ai/api/v2/custom_index/add/sync",
                        headers=headers,
                        json=data,
                        timeout=75,
                    )
                    if response.status_code != 200:
                        print(response.json())
                        print(f"ERROR: status code = {response.status_code}, current pk = {item['pk']}")
                        error = True
                    break
                except:
                    time.sleep(1)
                    attempt += 1
            if error or attempt == num_attempts:
                print(f"ERROR REPORTED: attempt = {attempt}")
                error = True
                break
        num_batches += 1
        if error:
            print("ERROR REPORTED: exiting")
            break
        res = q_iter.next()
    q_iter.close()



def batch_metadata_files() -> None:
    import json

    from openai import OpenAI


    basedir = "/Users/njc/Documents/programming/opb/data/courtlistener_bulk/"
    # opinion_filename = basedir + "opinions-2024-05-06.csv.bz2"
    # cluster_filename = basedir + "opinion-clusters-2024-05-06.csv.bz2"
    # docket_filename = basedir + "dockets-2024-05-06.csv.bz2"
    # people_filename = basedir + "people-db-people-2024-05-06.csv"
    # court_filename = basedir + "courts-2024-05-06.csv.bz2"
    # opinion_ids_filename = basedir + "opinion_ids"
    # cluster_ids_filename = basedir + "cluster_ids"
    # docket_ids_filename = basedir + "docket_ids"
    opinion_data_filename = basedir + "opinion_data"
    cluster_data_filename = basedir + "cluster_data"
    docket_data_filename = basedir + "docket_data"

    # docket_data = get_data_dictionary(docket_filename, docket_ids_filename, 100000, docket_row_data, court_filename=court_filename)
    # with pathlib.Path(docket_data_filename).open("w") as f:
    #     f.write(str(docket_data))
    # cluster_data = get_data_dictionary(cluster_filename, cluster_ids_filename, 100000, cluster_row_data)
    # with pathlib.Path(cluster_data_filename).open("w") as f:
    #     f.write(str(cluster_data))
    # opinion_data = get_data_dictionary(opinion_filename, opinion_ids_filename, 100000, opinion_row_data, people_filename=people_filename)
    # with pathlib.Path(opinion_data_filename).open("w") as f:
    #     f.write(str(opinion_data))

    with pathlib.Path(docket_data_filename).open("r") as f:
        docket_data = ast.literal_eval(f.read())
    with pathlib.Path(cluster_data_filename).open("r") as f:
        cluster_data = ast.literal_eval(f.read())
    with pathlib.Path(opinion_data_filename).open("r") as f:
        opinion_data = ast.literal_eval(f.read())

    client = OpenAI()
    openai_files = client.files.list()
    batches = client.batches.list()
    coll = Collection("courtlistener_bulk")
    for page in batches.iter_pages():
        for batch in page.data:
            metadatas, texts, vectors, chunk_idxs, opinion_ids = [], [], [], [], []
            if batch.status != "completed":
                continue

            input_file = next(
                (f for f in openai_files if batch.input_file_id == f.id),
                None,
            )
            if input_file is None:
                print("input file not found in API for " + batch.input_file_id)
                continue

            input_filename = input_file.filename
            print(input_filename)

            if not pathlib.Path(basedir + input_filename).exists():
                print("input file not found locally for " + batch.input_file_id)
                continue

            result_file_id = batch.output_file_id
            result_file_name = input_filename.split(".")[0] + "_out.jsonl"
            if not pathlib.Path(basedir + result_file_name).exists():
                result = client.files.content(result_file_id).content
                with pathlib.Path(basedir + result_file_name).open("wb") as f:
                    f.write(result)
            metadatas, texts, vectors, chunk_idxs, opinion_ids = [], [], [], [], []

            customid_inline = {}
            inline_text = {}
            with pathlib.Path(basedir + input_filename).open("r") as in_f:
                # index input lines
                for j, line in enumerate(in_f, start=1):
                    input_data = json.loads(line)
                    customid_inline[input_data["custom_id"]] = j
                    # just need the text from the input file
                    inline_text[j] = input_data["body"]["input"]
            with pathlib.Path(basedir + result_file_name).open("r") as out_f:
                for j, line in enumerate(out_f, start=1):
                    output_data = json.loads(line)
                    # check output
                    if output_data["response"]["status_code"] != 200:
                        print(f"error: bad status code for batch {batch.id} id {output_data['custom_id']}")
                        continue
                    # get vector
                    vector = output_data["response"]["body"]["data"][0]["embedding"]
                    # get text
                    inline = customid_inline[output_data["custom_id"]]
                    text = inline_text[inline]
                    # get metadata
                    custom_id_split = output_data["custom_id"].split("-")
                    cluster_id = int(custom_id_split[0])
                    opinion_id = int(custom_id_split[1])
                    chunk_idx = int(custom_id_split[2])
                    opinion_ids.append(opinion_id)
                    chunk_idxs.append(chunk_idx)
                    metadata = {}
                    metadata.update(cluster_data[cluster_id])
                    metadata.update(opinion_data[opinion_id])
                    metadata.update(docket_data[metadata["docket_id"]])
                    # add to batch
                    metadatas.append(metadata)
                    texts.append(text)
                    vectors.append(vector)
                    if len(metadatas) == 5000:
                        print(f"j = {j}")
                        data = [{
                            "vector": vectors[k],
                            "metadata": metadatas[k],
                            "text": texts[k],
                            "chunk_index": chunk_idxs[k],
                            "opinion_id": opinion_ids[k],
                        } for k in range(len(texts))]
                        upload_result = coll.insert(data)
                        if upload_result.insert_count != 5000:
                            print(f"error: bad upload, j = {j}")
                            continue
                        metadatas, texts, vectors, chunk_idxs, opinion_ids = [], [], [], [], []
            # upload the last <1000 lines
            if len(metadatas) > 0:
                data = [{
                    "vector": vectors[i],
                    "metadata": metadatas[i],
                    "text": texts[i],
                    "chunk_index": chunk_idxs[i],
                    "opinion_id": opinion_ids[i],
                } for i in range(len(texts))]
                upload_result = coll.insert(data)
                if upload_result.insert_count != len(metadatas):
                    print("error: bad upload for last batch in file")

def update_chunks() -> None:
    import json

    basedir = "/Users/njc/Documents/programming/opb/data/courtlistener_bulk/"
    opinion_data_filename = basedir + "opinion_data"
    cluster_data_filename = basedir + "cluster_data"
    docket_data_filename = basedir + "docket_data"

    with pathlib.Path(docket_data_filename).open("r") as f:
        docket_data = ast.literal_eval(f.read())
    with pathlib.Path(cluster_data_filename).open("r") as f:
        cluster_data = ast.literal_eval(f.read())
    with pathlib.Path(opinion_data_filename).open("r") as f:
        opinion_data = ast.literal_eval(f.read())

    q_iter = Collection("courtlistener").query_iterator(
        expr="",
        output_fields=["metadata"],
        batch_size=1000,
    )
    res = q_iter.next()
    while len(res) > 0:
        for hit in res:
            if "ai_summary" in hit["metadata"] and hit["metadata"]["id"] in opinion_data:
                opinion_data[hit["metadata"]["id"]]["ai_summary"] = hit["metadata"]["ai_summary"]
        res = q_iter.next()
    q_iter.close()
    coll_name = "test_firebase"
    coll = Collection(coll_name)
    for i in range(1, 286):
        metadatas, texts, vectors, chunk_idxs, opinion_ids = [], [], [], [], []
        source_ids = set()
        in_filename =  f"chunks_{i}.jsonl"
        out_filename = f"chunks_{i}_out.jsonl"
        customid_inline = {}
        inline_text = {}
        if not pathlib.Path(basedir + "completed/" + in_filename).exists():
            continue
        with pathlib.Path(basedir + "completed/" + in_filename).open("r") as in_f:
            print(in_filename)
            # index input lines
            for j, line in enumerate(in_f, start=1):
                input_data = json.loads(line)
                customid_inline[input_data["custom_id"]] = j
                # just need the text from the input file
                inline_text[j] = input_data["body"]["input"]
        with pathlib.Path(basedir + "completed/" + out_filename).open("r") as out_f:
            for j, line in enumerate(out_f, start=1):
                if i == 1 and j <= 10000:
                    continue
                output_data = json.loads(line)
                # get vector
                vector = output_data["response"]["body"]["data"][0]["embedding"]
                # get text
                inline = customid_inline[output_data["custom_id"]]
                text = inline_text[inline]
                # get metadata
                custom_id_split = output_data["custom_id"].split("-")
                cluster_id = int(custom_id_split[0])
                opinion_id = int(custom_id_split[1])
                chunk_idx = int(custom_id_split[2])
                opinion_ids.append(opinion_id)
                chunk_idxs.append(chunk_idx)
                metadata = {}
                metadata.update(cluster_data[cluster_id])
                metadata.update(opinion_data[opinion_id])
                metadata.update(docket_data[metadata["docket_id"]])
                # add to batch
                metadatas.append(metadata)
                texts.append(text)
                vectors.append(vector)
                if len(metadatas) == 10000:
                    print(f"j = {j}")
                    pks = [uuid.uuid1().int>>64 for _ in range(10000)]
                    data = [{
                        "pk": pks[k],
                        "vector": vectors[k],
                        "court_id": metadatas[k]["court_id"],
                        "date_filed": metadatas[k]["date_filed"],
                        "source_id": opinion_ids[k],
                    } for k in range(10000)]
                    upload_result = coll.insert(data)
                    if upload_result.insert_count != 10000:
                        print(f"error: bad upload, insert_count = {upload_result.insert_count}")
                        continue
                    for k in range(10000):
                        del metadatas[k]["id"]
                        if opinion_ids[k] not in source_ids:
                            source_ids.add(opinion_ids[k])
                            store_vdb_source(coll_name, opinion_ids[k], metadatas[k])
                        chunk_data = {"text": texts[k], "chunk_index": chunk_idxs[k]}
                        store_vdb_chunk(coll_name, opinion_ids[k], pks[k], chunk_data)
                    metadatas, texts, vectors, chunk_idxs, opinion_ids = [], [], [], [], []
        # upload the last <1000 lines
        if len(metadatas) > 0:
            data = [{
                "pk": uuid.uuid1().int>>64,
                "vector": vectors[j],
                "court_id": metadatas[j]["court_id"],
                "date_filed": metadatas[j]["date_filed"],
                "source_id": metadatas[j]["id"],
            } for j in range(len(texts))]
            upload_result = coll.insert(data)
            if upload_result.insert_count != len(metadatas):
                print("error: bad upload for last batch in file")
