"""Tests for CourtListener."""

def test_batchupload(capsys) -> None:
    from app.courtlistener import batch_metadata_files

    with capsys.disabled():
        batch_metadata_files()

def test_batchupdate():
    from app.courtlistener import update_chunks
    from app.milvusdb import DataType, FieldSchema, create_collection
    from app.models import EncoderParams, MilvusMetadataEnum

    opinion_id_field = FieldSchema("opinion_id", DataType.INT32, "The opinion ID")
    chunk_index_field = FieldSchema("chunk_index", DataType.INT32, "The chunk's index in the opinion")
    metadata_field = FieldSchema("metadata", DataType.JSON, "The chunk's metadata")

    create_collection("test_firebase", EncoderParams(dim=1536),
                      "Chunked opinions from CourtListener bulk data",
                      [opinion_id_field, chunk_index_field, metadata_field],
                        MilvusMetadataEnum.field,
                      )

    #update_chunks()

def test_batchretry():
    import pathlib

    from openai import OpenAI
    basedir = "data/courtlistener_bulk/"
    client = OpenAI()
    batches = client.batches.list()
    files = client.files.list()
    restarted_infile_ids = set()
    count = 0
    for page in batches.iter_pages():
        for batch in page.data:
            # handle completed batches
            # if batch.status != "completed":
            #     continue
            # print(f"batch {batch.id} completed")
            # in_exists, out_exists = False, False
            # for fpage in files.iter_pages():
            #     for f in fpage.data:
            #         if f.id == batch.input_file_id:
            #             print(f.filename)
            #             in_exists = True
            #         elif f.id == batch.output_file_id:
            #             print(f.filename)
            #             out_exists = True
            #     if in_exists and out_exists:
            #         break
            # if in_exists:
            #     print("deleted input file")
            #     client.files.delete(batch.input_file_id)
            # if out_exists:
            #     print("deleted output file")
            #     client.files.delete(batch.output_file_id)
            # if batch.error_file_id is not None:
            #     print("batch contains an error file, downloading")
            #     client.files.content(batch.error_file_id)
            #     error_file_info = client.files.retrieve(batch.error_file_id)
            #     error_filename = error_file_info.filename
            #     result = client.files.content(batch.error_file_id).content
            #     with pathlib.Path(basedir + error_filename).open("wb") as f:
            #         f.write(result)
            #     print(f"downloaded {error_filename}")

            # handle failed batches
            if batch.status != "failed":
                continue
            in_file_id = batch.input_file_id
            in_exists = False
            for fpage in files.iter_pages():
                for f in fpage.data:
                    if f.id == in_file_id:
                        print(f.filename)
                        in_exists = True
                        break
                if in_exists:
                    break
            if not in_exists:
                print(f"batch {batch.id} input file not found, skipping")
                continue
            if in_file_id in restarted_infile_ids:
                print(f"batch {batch.id} input file already restarted in another batch, skipping")
                continue
            print(f"batch {batch.id} failed and has input file, recreating")
            client.batches.create(
                completion_window="24h",
                endpoint="/v1/embeddings",
                input_file_id=in_file_id,
            )
            restarted_infile_ids.add(in_file_id)
            count += 1
            if count == 25:
                return

def test_batchdelete() -> None:
    import json
    import pathlib

    from app.milvusdb import delete_expr

    basedir = "data/courtlistener_bulk/completed/"
    with pathlib.Path(basedir + "chunks_121.jsonl").open("r") as f:
        lines = f.readlines()
    opinion_ids = set()
    for i, line in enumerate(lines, start=1):
        req = json.loads(line)
        custom_id_split = req["custom_id"].split("-")
        opinion_id = int(custom_id_split[1])
        if opinion_id not in opinion_ids:
            opinion_ids.add(opinion_id)
        if i % 1000 == 0:
            print(delete_expr("courtlistener_bulk", f"opinion_id in {list(opinion_ids)}"))
            opinion_ids = set()
    delete_expr("courtlistener_bulk", f"opinion_id in {list(opinion_ids)}")