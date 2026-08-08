"""
Aggregates research paper search results from Semantic Scholar, OpenAlex,
Crossref, and arXiv.
"""

import asyncio
import xml.etree.ElementTree as ET
from typing import List

import httpx

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import SearchResultItem

settings = get_settings()
logger = get_logger(__name__)

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _clean_title(title: str | None) -> str | None:
    if not title:
        return None

    title = title.strip()

    if title == "":
        return None

    if title.lower() == "untitled":
        return None

    return title


def _clean_authors(authors: List[str]) -> List[str]:
    cleaned = []

    for author in authors:
        if author and author.strip():
            cleaned.append(author.strip())

    return cleaned


def _build_paper_url(
    doi: str | None = None,
    fallback_url: str | None = None,
) -> str | None:
    if doi:
        return f"https://doi.org/{doi}"

    return fallback_url


def _normalize_title(title: str | None) -> str | None:
    if not title:
        return None
    import re
    return re.sub(r"[^a-z0-9]", "", title.lower())


def _normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    import re
    doi = doi.lower().strip()
    # Strip common version suffixes like .v1, /v1, -v1
    doi = re.sub(r"[./-]v\d+$", "", doi)
    return doi


async def _search_semantic_scholar(
    client: httpx.AsyncClient,
    topic: str,
    limit: int,
) -> List[SearchResultItem]:

    try:

        response = await client.get(
            f"{settings.SEMANTIC_SCHOLAR_BASE_URL}/paper/search",
            params={
                "query": topic,
                "limit": limit,
                "fields": (
                    "paperId,"
                    "title,"
                    "authors,"
                    "abstract,"
                    "year,"
                    "externalIds,"
                    "openAccessPdf"
                ),
            },
        )

        response.raise_for_status()

        data = response.json()

        results: List[SearchResultItem] = []

        for paper in data.get("data", []):

            title = _clean_title(paper.get("title"))

            if title is None:
                continue

            external_ids = paper.get("externalIds") or {}
            doi = external_ids.get("DOI")

            authors = _clean_authors(
                [
                    a.get("name", "")
                    for a in (paper.get("authors") or [])
                ]
            )

            open_access = paper.get("openAccessPdf") or {}

            paper_url = _build_paper_url(
                doi=doi,
                fallback_url=(
                    f"https://www.semanticscholar.org/paper/"
                    f"{paper.get('paperId')}"
                    if paper.get("paperId")
                    else None
                ),
            )

            results.append(
                SearchResultItem(
                    title=title,
                    authors=authors,
                    abstract=paper.get("abstract"),
                    year=paper.get("year"),
                    doi=doi,
                    paper_url=paper_url,
                    pdf_url=open_access.get("url"),
                    source="Semantic Scholar",
                )
            )

        return results

    except Exception as exc:

        logger.warning(
            "Semantic Scholar search failed: %s",exc,
        )

        return []


async def _search_openalex(
    client: httpx.AsyncClient,
    topic: str,
    limit: int,
) -> List[SearchResultItem]:

    try:

        response = await client.get(
            f"{settings.OPENALEX_BASE_URL}/works",
            params={
                "search": topic,
                "per_page": limit,
            },
        )

        response.raise_for_status()

        data = response.json()

        results: List[SearchResultItem] = []

        for work in data.get("results", []):

            title = _clean_title(work.get("title"))

            if title is None:
                continue

            authorships = work.get("authorships") or []

            authors = _clean_authors(
                [
                    author.get("author", {}).get("display_name", "")
                    for author in authorships
                ]
            )

            abstract = _reconstruct_openalex_abstract(
                work.get("abstract_inverted_index")
            )

            open_access = work.get("open_access") or {}

            doi = (work.get("doi") or "").replace(
                "https://doi.org/",
                "",
            )

            if doi == "":
                doi = None

            paper_url = _build_paper_url(
                doi=doi,
                fallback_url=work.get("id"),
            )

            results.append(
                SearchResultItem(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    year=work.get("publication_year"),
                    doi=doi,
                    paper_url=paper_url,
                    pdf_url=open_access.get("oa_url"),
                    source="OpenAlex",
                )
            )

        return results

    except Exception as exc:

        logger.warning(
            "OpenAlex search failed: %s",
            exc,
        )

        return []


def _reconstruct_openalex_abstract(inverted_index) -> str:
    if not inverted_index:
        return None
    position_map = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_map[pos] = word
    ordered = [position_map[i] for i in sorted(position_map.keys())]
    return " ".join(ordered) if ordered else None

async def _search_crossref(
    client: httpx.AsyncClient,
    topic: str,
    limit: int,
) -> List[SearchResultItem]:

    try:

        response = await client.get(
            f"{settings.CROSSREF_BASE_URL}/works",
            params={
                "query": topic,
                "rows": limit,
            },
        )

        response.raise_for_status()

        data = response.json()

        results: List[SearchResultItem] = []

        for item in data.get("message", {}).get("items", []):

            title_list = item.get("title") or []

            title = _clean_title(
                title_list[0] if title_list else None
            )

            # Skip papers with no valid title
            if title is None:
                continue

            authors = _clean_authors(
                [
                    f"{author.get('given','')} {author.get('family','')}".strip()
                    for author in (item.get("author") or [])
                ]
            )

            year = None

            date_parts = (
                item.get("published-print")
                or item.get("published-online")
                or {}
            ).get("date-parts")

            if date_parts and date_parts[0]:
                year = date_parts[0][0]

            doi = item.get("DOI")

            paper_url = _build_paper_url(
                doi=doi
            )

            pdf_url = None

            if item.get("link"):
                pdf_url = item["link"][0].get("URL")

            results.append(
                SearchResultItem(
                    title=title,
                    authors=authors,
                    abstract=item.get("abstract"),
                    year=year,
                    doi=doi,
                    paper_url=paper_url,
                    pdf_url=pdf_url,
                    source="Crossref",
                )
            )

        return results

    except Exception as exc:

        logger.warning(
            "Crossref search failed: %s",
            exc,
        )

        return []
async def _search_arxiv(
    client: httpx.AsyncClient,
    topic: str,
    limit: int,
) -> List[SearchResultItem]:

    try:

        response = await client.get(
            settings.ARXIV_BASE_URL,
            params={
                "search_query": f"all:{topic}",
                "start": 0,
                "max_results": limit,
            },
        )

        response.raise_for_status()

        root = ET.fromstring(response.text)

        results: List[SearchResultItem] = []

        for entry in root.findall("atom:entry", ARXIV_NS):

            title = _clean_title(
                entry.findtext(
                    "atom:title",
                    default="",
                    namespaces=ARXIV_NS,
                )
            )

            if title is None:
                continue

            summary = (
                entry.findtext(
                    "atom:summary",
                    default="",
                    namespaces=ARXIV_NS,
                )
                or ""
            ).strip()

            published = (
                entry.findtext(
                    "atom:published",
                    default="",
                    namespaces=ARXIV_NS,
                )
                or ""
            )

            year = (
                int(published[:4])
                if published[:4].isdigit()
                else None
            )

            authors = _clean_authors(
                [
                    (
                        author.findtext(
                            "atom:name",
                            default="",
                            namespaces=ARXIV_NS,
                        )
                        or ""
                    ).strip()
                    for author in entry.findall(
                        "atom:author",
                        ARXIV_NS,
                    )
                ]
            )

            pdf_url = None

            for link in entry.findall(
                "atom:link",
                ARXIV_NS,
            ):

                if (
                    link.attrib.get("title") == "pdf"
                    or link.attrib.get("type") == "application/pdf"
                ):
                    pdf_url = link.attrib.get("href")
                    break

            paper_url = entry.findtext(
                "atom:id",
                default="",
                namespaces=ARXIV_NS,
            )

            results.append(
                SearchResultItem(
                    title=title,
                    authors=authors,
                    abstract=summary,
                    year=year,
                    doi=None,
                    paper_url=paper_url,
                    pdf_url=pdf_url,
                    source="arXiv",
                )
            )

        return results

    except Exception as exc:

        logger.warning(
            "arXiv search failed: %s",
            exc,
        )

        return []
async def search_all_sources(
    topic: str,
    limit_per_source: int,
) -> List[SearchResultItem]:

    timeout = httpx.Timeout(
        settings.EXTERNAL_API_TIMEOUT_SECONDS
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        headers={
            "User-Agent": "AI-Research-Assistant/1.0"
        },
    ) as client:

        results_per_source = await asyncio.gather(

            _search_semantic_scholar(
                client,
                topic,
                limit_per_source,
            ),

            _search_openalex(
                client,
                topic,
                limit_per_source,
            ),

            _search_crossref(
                client,
                topic,
                limit_per_source,
            ),

            _search_arxiv(
                client,
                topic,
                limit_per_source,
            ),

            return_exceptions=True,
        )

    combined: List[SearchResultItem] = []

    for source in results_per_source:

        if isinstance(source, Exception):
            logger.warning(source)
            continue

        combined.extend(source)

    # -------------------------------
    # Remove duplicates
    # -------------------------------

    unique_papers: List[SearchResultItem] = []
    seen_titles = set()
    seen_dois = set()

    for paper in combined:
        norm_title = _normalize_title(paper.title)
        norm_doi = _normalize_doi(paper.doi)

        # Check if already seen by title or DOI
        is_duplicate = False
        if norm_title and norm_title in seen_titles:
            is_duplicate = True
        elif norm_doi and norm_doi in seen_dois:
            is_duplicate = True

        if not is_duplicate:
            if norm_title:
                seen_titles.add(norm_title)
            if norm_doi:
                seen_dois.add(norm_doi)
            unique_papers.append(paper)
        else:
            # Find the existing paper and merge metadata
            existing = None
            for p in unique_papers:
                if norm_title and _normalize_title(p.title) == norm_title:
                    existing = p
                    break
                if norm_doi and _normalize_doi(p.doi) == norm_doi:
                    existing = p
                    break

            if existing:
                # Merge metadata preferring richer content
                if (existing.abstract is None or len(existing.abstract.strip()) < 10) and paper.abstract:
                    existing.abstract = paper.abstract
                if existing.pdf_url is None and paper.pdf_url:
                    existing.pdf_url = paper.pdf_url
                if existing.paper_url is None and paper.paper_url:
                    existing.paper_url = paper.paper_url
                if existing.year is None and paper.year:
                    existing.year = paper.year
                if existing.doi is None and paper.doi:
                    existing.doi = paper.doi

    papers = unique_papers

    # -------------------------------
    # Sort papers
    # -------------------------------

    papers.sort(
        key=lambda p: (
            p.year or 0,
            len(p.abstract or "")
        ),
        reverse=True,
    )

    return papers
