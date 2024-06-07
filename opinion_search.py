"""Search court opinions using CAP and courtlistener data."""
from __future__ import annotations

from cap import cap
from courtlistener import courtlistener_search


def opinion_search(
    query: str,
    jurisdiction: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    k: int = 10,
) -> list[dict]:
    """Search CAP and courtlistener collections for relevant opinions.

    Parameters
    ----------
    query : str
        The users query
    jurisdiction : str | None
        The jurisdiction to filter by, by default None
    from_date : str | None
        The earliest date to filter by, by default None
    to_date : str | None
        The latest date to filter by, by default None
    k : int
        The number of results to return, by default 10

    Returns
    -------
    list[dict]
        A list of dicts containing the results from the search query

    """
    vdb_hits = []
    # check if jurisdiction is in CAP
    cap_juris = None
    if jurisdiction == "ar":
        cap_juris = "Ark."
    elif jurisdiction == "il":
        cap_juris = "Ill."
    if jurisdiction == "nc":
        cap_juris = "N.C."
    if jurisdiction == "nm":
        cap_juris = "N.M."
    if cap_juris:
        vdb_result = cap(query, k, cap_juris, from_date, to_date)
        vdb_hits = vdb_result["result"]
    # get courtlistener results
    search_result = courtlistener_search(query, k, jurisdiction, from_date, to_date)
    search_hits = search_result["result"]
    # get k closest results from either tool
    hits = []
    i, j = 0, 0
    while len(hits) < k:
        vdb_dist = vdb_hits[i]["distance"] if i < len(vdb_hits) else 1
        search_dist = search_hits[j]["distance"] if j < len(search_hits) else 1
        if vdb_dist <= search_dist:
            hits.append(vdb_hits[i])
            i += 1
        else:
            hits.append(search_hits[j])
            j += 1
    return hits
