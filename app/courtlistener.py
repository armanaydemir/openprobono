"""A module for interacting with the CourtListener API. Written by Arman Aydemir."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import requests
from langfuse.decorators import observe

from app.milvusdb import (
    fuzzy_keyword_query,
    get_expr,
    query,
    upload_courtlistener,
)

if TYPE_CHECKING:
    from app.models import OpinionSearchRequest

courtlistener_token = os.environ["COURTLISTENER_API_KEY"]
courtlistener_header = {"Authorization": "Token " + courtlistener_token}
base_url = "https://www.courtlistener.com/api/rest/v3"
courtlistener_collection = "courtlistener"
courtlistener_timeout = 10 #seconds

courtlistener_tool_args = {
    "jurisdiction": {
        "type": "string",
        "description": (
            "The two-letter abbreviation of a state or territory, e.g. 'NJ' or 'TX', "
            "to filter query results by jurisdiction. Use 'US' for federal courts."
        ),
    },
    "keyword-qr": {
        "type": "string",
        "description": (
            "A keyword query to search for exact names and terms."
        ),
    },
    "after-date": {
        "type": "string",
        "description": (
            "The after date for the query date range in YYYY-MM-DD "
            "format."
        ),
    },
    "before-date": {
        "type": "string",
        "description": (
            "The before date for the query date range in YYYY-MM-DD "
            "format."
        ),
    },
}

# https://github.com/freelawproject/courtlistener/discussions/3114
# manual mapping from two-letter state abbreviations to courtlistener court_id format
jurisdiction_codes = {
    "us-app": "ca1 ca2 ca3 ca4 ca5 ca6 ca7 ca8 ca9 ca10 ca11 cadc cafc",
    "us-dis": (
        "dcd almd alnd alsd akd azd ared arwd cacd caed cand casd cod ctd ded flmd "
        "flnd flsd gamd gand gasd hid idd ilcd ilnd ilsd innd insd iand iasd ksd kyed "
        "kywd laed lamd lawd med mdd mad mied miwd mnd msnd mssd moed mowd mtd ned nvd "
        "nhd njd nmd nyed nynd nysd nywd nced ncmd ncwd ndd ohnd ohsd oked oknd okwd "
        "ord paed pamd pawd rid scd sdd tned tnmd tnwd txed txnd txsd txwd utd vtd "
        "vaed vawd waed wawd wvnd wvsd wied wiwd wyd gud nmid prd vid californiad "
        "illinoised illinoisd indianad orld ohiod pennsylvaniad southcarolinaed "
        "southcarolinawd tennessed canalzoned"
    ),
    "us-sup": "scotus",
    "us-misc": (
        "bap1 bap2 bap6 bap8 bap9 bap10 ag afcca asbca armfor acca uscfc tax bia olc "
        "mc mspb nmcca cavc bva fiscr fisc cit usjc jpml cc com ccpa cusc bta eca "
        "tecoa reglrailreorgct kingsbench"
    ),
    "al": "almd alnd alsd almb alnb alsb ala alactapp alacrimapp alacivapp",
    "ak": "akd akb alaska alaskactapp",
    "az": "azd arb ariz arizctapp ariztaxct",
    "ar": "ared arwd areb arwb ark arkctapp arkworkcompcom arkag",
    "as": "amsamoa amsamoatc",
    "ca": (
        "cacd caed cand casd californiad cacb caeb canb casb cal calctapp "
        "calappdeptsuper calag"
    ),
    "co": "cod cob colo coloctapp coloworkcompcom coloag",
    "ct": "ctd ctb conn connappct connsuperct connworkcompcom",
    "de": "ded deb del delch delorphct delsuperct delctcompl delfamct deljudct",
    "dc": "dcd dcb dc",
    "fl": "flmd flnd flsd flmb flnb flsb fla fladistctapp flaag",
    "ga": "gamd gand gasd gamb ganb gasb ga gactapp",
    "gu": "gud gub",
    "hi": "hid hib haw hawapp",
    "id": "idd idb idaho idahoctapp",
    "il": "ilcd ilnd ilsd illinoised illinoisd ilcb ilnb ilsb ill illappct",
    "in": "innd insd indianad innb insb ind indctapp indtc",
    "ia": "iand iasd ianb iasb iowa iowactapp",
    "ks": "ksd ksb kan kanctapp kanag",
    "ky": "kyed kywd kyeb kywb ky kyctapp kyctapphigh",
    "la": "laed lamd lawd laeb lamb lawb la lactapp laag",
    "me": "med bapme meb me mesuperct",
    "md": "mdd mdb md mdctspecapp mdag",
    "ma": "mad bapma mab mass massappct masssuperct massdistct masslandct maworkcompcom",
    "mi": "mied miwd mieb miwb mich michctapp",
    "mn": "mnd mnb minn minnctapp minnag",
    "ms": "msnd mssd msnb mssb miss missctapp",
    "mo": "moed mowd moeb mowb mo moctapp moag",
    "mt": "mtd mtb mont monttc montag",
    "ne": "ned nebraskab neb nebctapp nebag",
    "nv": "nvd nvb nev nevapp",
    "nh": "nhd nhb nh",
    "nj": "njd njb nj njsuperctappdiv njtaxct njch",
    "nm": "nmd nmb nm nmctapp",
    "ny": (
        "nyed nynd nysd nywd nyeb nynb nysb nywb ny nyappdiv nyappterm nysupct "
        "nycountyctny nydistct nyjustct nyfamct nysurct nycivct nycrimct nyag"
    ),
    "nc": "nced ncmd ncwd nceb ncmb ncwb nc ncctapp ncsuperct ncworkcompcom",
    "nd": "ndd ndb nd ndctapp",
    "mp": "nmariana cnmisuperct cnmitrialct",
    "oh": "ohnd ohsd ohiod ohnb ohsb ohio ohioctapp ohioctcl",
    "ok": (
        "oked oknd okwd okeb oknb okwb okla oklacivapp oklacrimapp oklajeap "
        "oklacoj oklaag"
    ),
    "or": "ord orb or orctapp ortc",
    "pa": "paed pamd pawd pennsylvaniad paeb pamb pawb pa pasuperct pacommwct cjdpa",
    "pr": "prsupreme prapp prd prb",
    "ri": "rid rib ri risuperct",
    "sc": "scd southcarolinaed southcarolinawd scb sc scctapp",
    "sd": "sdd sdb sd",
    "tn": (
        "tned tnmd tnwd tennessed tneb tnmb tnwb tennesseeb tenn tennctapp "
        "tenncrimapp tennworkcompcl tennworkcompapp tennsuperct"
    ),
    "tx": (
        "txed txnd txsd txwd txeb txnb txsb txwb tex texapp texcrimapp "
        "texreview texjpml texag"
    ),
    "ut": "utd utb utah utahctapp",
    "vt": "vtd vtb vt vtsuperct",
    "va": "vaed vawd vaeb vawb va vactapp",
    "vi": "vid vib",
    "wa": "waed wawd waeb wawb wash washctapp washag washterr",
    "wv": "wvnd wvsd wvnb wvsb wva",
    "wi": "wied wiwd wieb wiwb wis wisctapp wisag",
    "wy": "wyd wyb wyo",
}


def search(q: str) -> dict:
    """Call the general search api from courtlistener.

    Parameters
    ----------
    q : str
        the query

    Returns
    -------
    dict
        dict containing the results

    """
    http_code_ok = 200
    search_url = base_url + "/search/?q="
    max_retries = 5
    retry_delay = 1 # second
    for _ in range(max_retries):
        try:
            response = requests.get(
                search_url + q,
                headers=courtlistener_header,
                timeout=courtlistener_timeout,
            )
            if response.status_code == http_code_ok:
                return response.json()
        except requests.exceptions.ReadTimeout:  # noqa: PERF203
            time.sleep(retry_delay)
    return {}

def get_target(target_id: int, target_name: str) -> dict:
    """Get a courtlistener target object by id.

    Parameters
    ----------
    target_id : int
        The target object's id
    target_name : str
        The target object's name. Must be one of:
        "opinion", "cluster", "docket", "court", "people"

    Returns
    -------
    dict
        The target object with the matching id

    """
    url = None
    if target_name == "opinion":
        url = base_url + "/opinions/?id="
    elif target_name == "cluster":
        url = base_url + "/clusters/?id="
    elif target_name == "docket":
        url = base_url + "/dockets/?id="
    elif target_name == "court":
        url = base_url + "/courts/?id="
    elif target_name == "people":
        url = base_url + "/people/?id="
    if url is None:
        return {}
    response = requests.get(
        url + str(target_id),
        headers=courtlistener_header,
        timeout=courtlistener_timeout,
    )
    return response.json()["results"][0]

def get_opinion(result:dict) -> dict:
    """Get the full opinion info for a search result from search().

    Parameters
    ----------
    result : dict
        a single result from search()

    Returns
    -------
    dict
        dict containing the Opinion info

    """
    # lookup the opinion
    op = get_target(result["id"], "opinion")
    # getting the text, in the best format possible
    op["text"] = op["xml_harvard"]
    # these are backup formats
    backups = ["html_with_citations", "html", "plain_text", "html_lawbox", "html_columbia"]
    b_index = 0
    while op["text"] == "" and b_index < len(backups):
        op["text"] = op[backups[b_index]]
        b_index += 1
    return op

def get_search_date(date_plus_timerange: str) -> str:
    split_date = date_plus_timerange.split("T")
    date_filed = split_date[0]
    split_time = split_date[1].split("-")
    len_split = 2
    if len(split_time) == len_split and split_time[0] > split_time[1]:
        # the time range goes overnight, so add one day to the date
        dt = datetime.strptime(date_filed, "%Y-%m-%d").replace(tzinfo=UTC)
        dt += timedelta(days=1)
        date_filed = dt.strftime("%Y-%m-%d")
    return date_filed

def get_case_name(result: dict) -> str:
    if result["caseName"]:
        return result["caseName"]
    # cluster level data: case name
    cluster = get_target(result["cluster_id"], "cluster")
    # prefer short name to full name
    if cluster["case_name"]:
        return cluster["case_name"]
    if cluster["case_name_short"]:
        return cluster["case_name_short"]
    return cluster["case_name_full"]

def get_author_name(result: dict) -> str:
    # person level data: author name
    author = get_target(result["author_id"], "people")
    full_name = ""
    if author["name_first"]:
        full_name += author["name_first"]
    if author["name_middle"]:
        full_name += (" " if full_name else "") + author["name_middle"]
    if author["name_last"]:
        full_name += (" " if full_name else "") + author["name_last"]
    return full_name

def upload_search_result(result: dict) -> None:
    # check if the opinion is already in the collection
    expr = f"opinion_id=={result['id']}"
    hits = get_expr(courtlistener_collection, expr)
    if hits["result"] and len(hits["result"]) > 0:
        return
    opinion = get_opinion(result)
    opinion["cluster_id"] = result["cluster_id"]
    opinion["court_id"] = result["court_id"]
    opinion["date_filed"] = get_search_date(result["dateFiled"])
    opinion["docket_id"] = result["docket_id"]
    opinion["court_name"] = result["court"]
    opinion["citations"] = result["citation"]
    opinion["case_name"] = get_case_name(result)
    if result["author_id"]:
        opinion["author_id"] = result["author_id"]
        opinion["author_name"] = get_author_name(result)
    upload_courtlistener(courtlistener_collection, opinion)

@observe(capture_output=False)
def courtlistener_search(request: OpinionSearchRequest) -> dict:
    """Search courtlistener for a query.

    Search, get opinion text, upload the opinion data to milvus, and query it.

    Parameters
    ----------
    request : OpinionSearchRequest
        The opinion search request object

    Returns
    -------
    dict
        the response with relevant info from courtlistener

    """
    # use semantic query if keyword not given
    if request.keyword_query:
        query_str = '"' + request.keyword_query + '"'
    else:
        query_str = request.query
    # add options to query string
    valid_jurisdics = []
    if request.jurisdictions:
        # look up each str in dictionary, append matches as lists
        for juris in request.jurisdictions:
            if juris in jurisdiction_codes:
                valid_jurisdics += jurisdiction_codes[juris].split(" ")
        # clear duplicate federal district jurisdictions if they exist
        valid_jurisdics = list(set(valid_jurisdics))
        query_str += f"&court={' '.join(valid_jurisdics)}"
    if request.after_date:
        dt = request.after_date.split("-")
        # needs to be in MM-DD-YYYY format
        query_str += f"&filed_after={dt[1]}-{dt[2]}-{dt[0]}"
    if request.before_date:
        dt = request.before_date.split("-")
        # needs to be in MM-DD-YYYY format
        query_str += f"&filed_before={dt[1]}-{dt[2]}-{dt[0]}"

    opinion_ids = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for result in search(query_str)["results"][:request.k]:
            opinion_ids.append(result["id"])
            ctx = copy_context()
            def task(r=result, context=ctx):  # noqa: ANN001, ANN202
                return context.run(upload_search_result, r)
            futures.append(executor.submit(task))

        for future in as_completed(futures):
            _ = future.result()
    if not request.keyword_query:
        opinion_ids = None
    return courtlistener_query(request, opinion_ids)

@observe(capture_input=False, capture_output=False)
def courtlistener_query(
    request: OpinionSearchRequest,
    search_result_ids: list[str] | None = None,
) -> dict:
    """Query Courtlistener data.

    Parameters
    ----------
    request : OpinionSearchRequest
        The opinion search request object
    search_result_ids : list[str] | None = None
        The opinion ids from CL search results if keyword_query was used,
        by default None

    Returns
    -------
    dict
        Contains `message`, `result` list if successful

    """
    expr = ""
    # copy keyword query to semantic if not given
    if request.jurisdictions:
        expr = f"metadata['court_id'] in {request.jurisdictions}"
    if request.after_date:
        expr += (" and " if expr else "") + f"metadata['date_filed']>'{request.after_date}'"
    if request.before_date:
        expr += (" and " if expr else "") + f"metadata['date_filed']<'{request.before_date}'"
    if request.keyword_query:
        keyword_query = fuzzy_keyword_query(request.keyword_query)
        expr += (" and " if expr else "")
        keyword_expr = f"text like '% {keyword_query} %'"
        if search_result_ids:
            expr += f"(metadata['id'] in {search_result_ids} or " + keyword_expr + ")"
        else:
            expr += keyword_expr
    return query(courtlistener_collection, request.query, request.k, expr)
