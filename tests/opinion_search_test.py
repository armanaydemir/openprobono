"""Tests for opinion search."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from app import main
from app.courtlistener import jurisdiction_codes
from app.models import OpinionSearchRequest
from app.opinion_search import add_opinion_summary, opinion_search

if TYPE_CHECKING:
    import requests

client = TestClient(main.api, headers={"X-API-KEY": os.environ["OPB_TEST_API_KEY"]})

test_query = "tow truck"
test_jurisdictions = ["nc"]
test_keyword_query = "tow truck"
test_after_date = "1989-12-31"
test_before_date = "2000-01-01"

def _test_results(results: list) -> None:
    """Test opinion search query results."""
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], dict)
    assert "distance" in results[0]
    assert "entity" in results[0]
    assert isinstance(results[0]["entity"], dict)
    assert "metadata" in results[0]["entity"]
    assert isinstance(results[0]["entity"]["metadata"], dict)
    assert "text" in results[0]["entity"]
    assert isinstance(results[0]["entity"]["text"], str)

def _test_response(response: requests.Response) -> None:
    response_ok = 200
    assert response.status_code == response_ok
    response_json = response.json()
    assert response_json["message"] == "Success"
    assert "results" in response_json
    _test_results(response_json["results"])

@pytest.mark.parametrize(("query", "jurisdictions"), [(test_query, test_jurisdictions)])
def test_opinion_search(query: str, jurisdictions: list[str]) -> None:
    """Run the opinion_search function directly with jurisdictions."""
    request = OpinionSearchRequest(query=query, k=1, jurisdictions=jurisdictions)
    results = opinion_search(request)
    _test_results(results)
    query_juris_codes = []
    for juris in jurisdictions:
        query_juris_codes += jurisdiction_codes[juris].split(" ")
    assert results[0]["entity"]["metadata"]["court_id"] in query_juris_codes

@pytest.mark.parametrize("query", [test_query])
def test_query_unfiltered(query: str) -> None:
    """Run a query without jurisdiction and date filters."""
    params = {
        "query": query,
        "k": 1,
        "jurisdictions": None,
        "keyword_query": None,
        "after_date": None,
        "before_date": None,
    }
    response = client.post("/search_opinions", json=params)
    _test_response(response)

@pytest.mark.parametrize(("query", "jurisdictions"), [(test_query, test_jurisdictions)])
def test_query_jurisdiction(query: str, jurisdictions: list[str]) -> None:
    """Run a query with a jurisdiction filter."""
    params = {
        "query": query,
        "k": 1,
        "jurisdictions": jurisdictions,
        "keyword_query": None,
        "after_date": None,
        "before_date": None,
    }
    response = client.post("/search_opinions", json=params)
    _test_response(response)
    query_juris_codes = []
    for juris in jurisdictions:
        query_juris_codes += jurisdiction_codes[juris].split(" ")
    results = response.json()["results"]
    assert "court_id" in results[0]["entity"]["metadata"]
    assert results[0]["entity"]["metadata"]["court_id"] in query_juris_codes

@pytest.mark.parametrize(("query", "keyword_query"), [(test_query, test_keyword_query)])
def test_query_keyword(query: str, keyword_query: str) -> None:
    """Run a query with a keyword query filter."""
    params = {
        "query": query,
        "k": 1,
        "jurisdictions": None,
        "keyword_query": keyword_query,
        "after_date": None,
        "before_date": None,
    }
    response = client.post("/search_opinions", json=params)
    _test_response(response)
    results = response.json()["results"]
    assert keyword_query in results[0]["entity"]["text"]

@pytest.mark.parametrize(("query", "after_date"), [(test_query, test_after_date)])
def test_query_after_date(query: str, after_date: str) -> None:
    """Run a query with an after date filter."""
    params = {
        "query": query,
        "k": 1,
        "jurisdictions": None,
        "keyword_query": None,
        "after_date": after_date,
        "before_date": None,
    }
    response = client.post("/search_opinions", json=params)
    _test_response(response)
    results = response.json()["results"]
    assert "date_filed" in results[0]["entity"]["metadata"]
    assert results[0]["entity"]["metadata"]["date_filed"] > after_date

@pytest.mark.parametrize(("query", "before_date"), [(test_query, test_before_date)])
def test_query_before_date(query: str, before_date: str) -> None:
    """Run a query with a before date filter."""
    params = {
        "query": query,
        "k": 1,
        "jurisdictions": None,
        "keyword_query": None,
        "after_date": None,
        "before_date": before_date,
    }
    response = client.post("/search_opinions", json=params)
    _test_response(response)
    results = response.json()["results"]
    assert "date_filed" in results[0]["entity"]["metadata"]
    assert results[0]["entity"]["metadata"]["date_filed"] < before_date

@pytest.mark.parametrize(
    ("query", "jurisdictions", "keyword_query", "after_date", "before_date"),
    [(test_query, test_jurisdictions, test_keyword_query, test_after_date, test_before_date)],
)
def test_query_filtered(
    query: str,
    jurisdictions: list[str],
    keyword_query: str,
    after_date: str,
    before_date: str,
) -> None:
    """Run a query with all filters enabled."""
    params = {
        "query": query,
        "k": 1,
        "jurisdictions": jurisdictions,
        "keyword_query": keyword_query,
        "after_date": after_date,
        "before_date": before_date,
    }
    response = client.post("/search_opinions", json=params)
    _test_response(response)
    results = response.json()["results"]
    query_juris_codes = []
    for juris in jurisdictions:
        query_juris_codes += jurisdiction_codes[juris].split(" ")
    assert "court_id" in results[0]["entity"]["metadata"]
    assert results[0]["entity"]["metadata"]["court_id"] in query_juris_codes
    assert keyword_query in results[0]["entity"]["text"]
    assert "date_filed" in results[0]["entity"]["metadata"]
    assert results[0]["entity"]["metadata"]["date_filed"] < before_date
    assert results[0]["entity"]["metadata"]["date_filed"] > after_date

def test_summary() -> None:
    pass
