"""Classes for creating/loading knowledge bases, primarily for evaluation."""
from __future__ import annotations

import os
from abc import abstractmethod
from pathlib import Path
from typing import Generator, Protocol

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, ElementMetadata, Text
from unstructured.partition.auto import partition

from app.encoders import embed_strs
from app.milvusdb import get_expr, load_vdb_param, upload_data


class KnowledgeBase(Protocol):
    """Base class for evaluation data."""

    @abstractmethod
    def generate_elements(self: KnowledgeBase) -> Generator[tuple[str, list[Element]]]:
        """Generate elements from some configured sources.

        Parameters
        ----------
        self : KnowledgeBase
            The current instance of KnowledgeBase.

        Yields
        ------
        Generator[tuple[str, list[Element]]]
            source, elements

        """


    def populate_database(
        self: KnowledgeBase,
        collection_name: str,
        chunk_hardmax: int,
        chunk_softmax: int,
        overlap: int,
    ) -> bool:
        """Populate a database for the KnowledgeBase instance.

        Parameters
        ----------
        self : KnowledgeBase
            The current instance of KnowledgeBase.
        collection_name : str
            The name of the milvus Collection to populate.
        chunk_hardmax : int
            The maximum number of characters to use when chunking.
        chunk_softmax : int
            The preferred number of characters to use when chunking.
        overlap : int
            The number of characters to overlap when chunking.

        Returns
        -------
        bool
            Whether or not the database was populated successfully.

        """
        encoder = load_vdb_param(collection_name, "encoder")
        for _, elements in self.generate_elements():
            chunks = chunk_by_title(
                elements,
                max_characters=chunk_hardmax,
                new_after_n_chars=chunk_softmax,
                overlap=overlap,
            )
            num_chunks = len(chunks)
            texts, metadatas = [], []
            for i in range(num_chunks):
                texts.append(chunks[i].text)
                metadatas.append(chunks[i].metadata.to_dict())
            vectors = embed_strs(texts, encoder)
            data = [{
                "vector": vectors[i],
                "metadata": metadatas[i],
                "text": texts[i],
            } for i in range(len(texts))]
            result = upload_data(collection_name, data)
            if result["message"] != "Success":
                return False
        return True


    def generate_vdb_elements(
        self: KnowledgeBase,
        collection_name: str,
        source: str,
    ) -> Generator[list[Element]]:
        """Load the elements for a given source from a milvus Collection.

        Parameters
        ----------
        self : KnowledgeBase
            The current instance of KnowledgeBase.
        collection_name : str
            The name of the milvus Collection.
        source : str
            The source to load elements from.

        Returns
        -------
        Generator[list[Element]]
            A list of elements from the source found in the milvus Collection.

        """
        expr = f"metadata['url']=='{source}'"
        hits = get_expr(collection_name, expr)["result"]
        for hit in hits:
            del hit["vector"]
        return [
            Text(
                text=hit["text"],
                metadata=ElementMetadata(
                    url=source,
                    page_number=hit["metadata"]["page_number"],
                ),
            ) for hit in hits
        ]


class KnowledgeBaseNC(KnowledgeBase):
    """Evaluation data for the NC General Statutes."""

    def __init__(self, source_list_file: str) -> None:
        """Load a list of sources."""
        if Path(source_list_file).exists():
            with Path(source_list_file).open() as f:
                self.sources = [line.strip() for line in f]


    def generate_elements(
        self: KnowledgeBaseNC,
    ) -> Generator[tuple[str, list[Element]]]:
        """Generate a list of elements from their source URLs.

        Parameters
        ----------
        self : KnowledgeBaseNC
            The current instance of KnowledgeBaseNC.

        Yields
        ------
        Generator[tuple[str, list[Element]]]
            The elements generated from a chapter.

        """
        for chapter_pdf_url in self.sources:
            elements = partition(url=chapter_pdf_url, content_type="application/pdf")
            yield chapter_pdf_url, elements

    def populate_database(
        self: KnowledgeBase,
        collection_name: str,
        chunk_hardmax: int,
        chunk_softmax: int,
        overlap: int,
    ) -> bool:
        return super().populate_database(
            collection_name,
            chunk_hardmax,
            chunk_softmax,
            overlap,
        )

    def load_elements(
        self: KnowledgeBaseNC,
        collection_name: str,
    ) -> Generator[list[Element]]:
        """Load elements from the milvus Collection.

        Parameters
        ----------
        self : KnowledgeBaseNC
            The current instance of KnowledgeBaseNC.
        collection_name : str
            The name of the milvus Collection to gather documents from.

        Yields
        ------
        Generator[list[Element]]
            The elements loaded from the milvus Collection.

        """
        for chapter_pdf_url in self.sources:
            yield self.generate_vdb_elements(collection_name, chapter_pdf_url)
