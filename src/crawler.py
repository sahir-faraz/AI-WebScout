
"""Respectful, relevance-focused website crawler for AI-WebScout."""
import traceback
from collections import deque
from urllib.parse import (
    urljoin, urlparse, urldefrag, parse_qsl, urlencode
)
from urllib.robotparser import RobotFileParser
import hashlib
import ipaddress
import socket
import time

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


USER_AGENT = "AI-WebScout/1.1 (website research prototype)"
TIMEOUT = 12
MAX_RESPONSE_BYTES = 5_000_000
MAX_LINKS_PER_PAGE = 150

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term",
    "utm_content", "gclid", "fbclid", "mc_cid", "mc_eid",
}

DROP_TAGS = [
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "form", "button", "svg", "iframe", "canvas",
    "template", "input", "select", "textarea", "dialog",
]

BOILERPLATE_WORDS = (
    "cookie", "consent", "breadcrumb", "sidebar", "navigation",
    "advert", "social-share", "share-buttons", "newsletter",
    "related-posts", "comments", "site-footer", "site-header",
)

CONTENT_SELECTORS = [
    "article",
    "main",
    '[role="main"]',
    ".document",
    ".body",
    ".content",
    ".section",
]


def _public_http_url(url):
    """Validate a public HTTP(S) URL and reject private IP addresses."""
    parsed = urlparse(url.strip())

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Enter a valid public HTTP or HTTPS URL.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing usernames or passwords are unsupported.")

    try:
        print(f"[DEBUG] Hostname being resolved: {parsed.hostname}")
        print(f"[DEBUG] Full URL being validated: {url}")
        
        infos = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )
        addresses = {ipaddress.ip_address(info[4][0]) for info in infos}

        if not addresses or any(not address.is_global for address in addresses):
            raise ValueError("Private and non-public addresses are not allowed.")

    except socket.gaierror as exc:
        raise ValueError("Website hostname could not be resolved.") from exc

    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    return parsed._replace(
        netloc=host + (f":{parsed.port}" if parsed.port else ""),
        fragment="",
    ).geturl()


def _normalise(url):
    """Remove fragments and common tracking parameters."""
    clean, _ = urldefrag(url.strip())
    parsed = urlparse(clean)

    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
        and not key.lower().startswith("utm_")
    ]

    # Preserve trailing slashes: some servers redirect /path to /path/.
    # Stripping the slash can create a redirect loop when redirects are disabled.
    return parsed._replace(
        path=parsed.path or "/",
        query=urlencode(query, doseq=True),
    ).geturl()


def _same_host(url, base_host):
    try:
        return (
            (urlparse(url).hostname or "").lower()
            == (base_host or "").lower()
        )
    except Exception:
        return False



def _robots_allowed(url, session=None):
    from urllib.parse import urlparse
    from urllib.robotparser import RobotFileParser
    import requests

    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    try:
        response = requests.get(
            robots_url,
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT,
            allow_redirects=False,
        )

        print(
            f"[ROBOTS] URL={robots_url} "
            f"STATUS={response.status_code}"
        )

        if response.status_code == 404:
            return True

        if response.status_code != 200:
            print(
                f"[ROBOTS] Access rules unavailable: "
                f"HTTP {response.status_code}"
            )
            return False

        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(response.text.splitlines())

        allowed = parser.can_fetch(USER_AGENT, url)
        print(f"[ROBOTS] Page allowed={allowed}")
        return allowed

    except requests.RequestException as exc:
        print(f"[ROBOTS] Request error: {exc}")
        return False



def _node_is_boilerplate(node):
    """Identify common navigation and non-content elements."""
    if node is None:
        return False

    attrs = getattr(node, "attrs", None)
    if not isinstance(attrs, dict):
        return False

    values = []
    for key in ("id", "class", "role", "aria-label"):
        value = attrs.get(key, "")
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))

    marker = " ".join(values).lower()
    return any(word in marker for word in BOILERPLATE_WORDS)


def _relevance_scores(texts, question):
    """Calculate TF-IDF similarity between a question and content blocks."""
    if not texts:
        return []

    if not question or not question.strip():
        return [1.0] * len(texts)

    try:
        matrix = TfidfVectorizer(
            stop_words="english",
            max_features=12000,
            ngram_range=(1, 2),
        ).fit_transform([question] + texts)

        return cosine_similarity(
            matrix[0:1], matrix[1:]
        ).flatten().tolist()

    except ValueError:
        return [0.0] * len(texts)


def _select_main_content(soup):
    """Prefer the main article or document instead of the entire page."""
    for selector in CONTENT_SELECTORS:
        candidates = soup.select(selector)

        if candidates:
            # Choose the candidate with the most readable text.
            return max(
                candidates,
                key=lambda node: len(node.get_text(" ", strip=True)),
            )

    return soup.body or soup


def _clean_page(html, page_url, question="", max_text_chars=18000):
    """
    Extract structured article text and relevant images.

    Output text preserves headings, paragraphs, lists, code blocks and tables.
    """
    soup = BeautifulSoup(html, "html.parser")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title else ""
    )

    canonical = soup.find(
        "link",
        rel=lambda value: value and "canonical" in value,
    )

    canonical_url = (
        urljoin(page_url, canonical.get("href", ""))
        if canonical and canonical.get("href")
        else page_url
    )

    # Remove non-content elements.
    for node in list(soup.find_all(DROP_TAGS)):
        node.decompose()

    for node in list(soup.find_all(attrs={"aria-hidden": "true"})):
        node.decompose()

    for node in list(soup.find_all(style=True)):
        style = node.get("style", "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            node.decompose()

    for node in list(soup.find_all(True)):
        if _node_is_boilerplate(node):
            node.decompose()

    main = _select_main_content(soup)

    # Extract blocks in document order, including code and table content.
    candidates = []
    block_tags = [
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "li", "blockquote", "pre", "table", "figcaption",
        "dt", "dd",
    ]

    for node in main.find_all(block_tags):
        if node.name == "li" and node.find_parent("li"):
            continue

        if node.name == "table":
            rows = []
            for row in node.find_all("tr"):
                cells = [
                    " ".join(cell.stripped_strings)
                    for cell in row.find_all(["th", "td"], recursive=False)
                ]
                if cells:
                    rows.append(" | ".join(cells))

            value = "\n".join(rows)
            kind = "table"

        elif node.name == "pre":
            value = node.get_text("\n", strip=True)
            kind = "code"

        else:
            value = " ".join(node.get_text(" ", strip=True).split())
            kind = node.name

        if not value:
            continue

        # Retain short headings and list items; they can carry useful context.
        if kind not in ("code", "table") and len(value) < 20:
            if kind not in ("h1", "h2", "h3", "h4", "h5", "h6", "li"):
                continue

        candidates.append({"kind": kind, "text": value})

    # Fallback: structured tags may be absent on some websites, or the
    # selected main container may contain only a short summary. If extraction
    # is empty or suspiciously short, try the full cleaned body text.
    structured_length = sum(len(item["text"]) for item in candidates)
    body = soup.body or soup
    fallback_text = body.get_text("\n", strip=True)
    fallback_text = "\n".join(
        line.strip() for line in fallback_text.splitlines() if line.strip()
    )

    if fallback_text and (
        not candidates or
        (structured_length < 200 and len(fallback_text) > structured_length + 100)
    ):
        candidates = [{"kind": "fallback", "text": fallback_text}]

    # Add useful page metadata when the visible body is sparse. Many public sites
    # expose a description or article summary in standard metadata tags.
    if sum(len(item["text"]) for item in candidates) < 500:
        meta_values = []
        for attrs in (
            {"name": "description"},
            {"property": "og:description"},
            {"name": "twitter:description"},
        ):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                value = " ".join(tag["content"].split())
                if value and value not in meta_values:
                    meta_values.append(value)
        if meta_values:
            candidates.extend({"kind": "p", "text": value} for value in meta_values)

    # Use JSON-LD article text/description as a last-resort source on sites that
    # publish structured metadata but little directly readable HTML.
    if sum(len(item["text"]) for item in candidates) < 300:
        import json
        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = script.string or script.get_text()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except (ValueError, TypeError):
                continue
            stack = payload if isinstance(payload, list) else [payload]
            while stack:
                obj = stack.pop(0)
                if isinstance(obj, dict):
                    for key in ("articleBody", "description", "headline", "name"):
                        value = obj.get(key)
                        if isinstance(value, str) and value.strip():
                            candidates.append({"kind": "p", "text": " ".join(value.split())})
                    for value in obj.values():
                        if isinstance(value, (dict, list)):
                            stack.extend(value if isinstance(value, list) else [value])
                elif isinstance(obj, list):
                    stack.extend(obj)

    # Remove duplicate text blocks without destroying the reading order.
    unique = []
    fingerprints = set()

    for item in candidates:
        fingerprint = " ".join(item["text"].lower().split())

        if fingerprint in fingerprints:
            continue

        fingerprints.add(fingerprint)
        unique.append(item)

    scores = _relevance_scores(
        [item["text"] for item in unique],
        question,
    )

    for item, score in zip(unique, scores):
        item["relevance"] = float(score)

    # Keep content in document order. Page-level relevance is used to rank
    # pages, not to discard paragraphs that provide important context.
    selected = unique

    # Construct readable, structured text within the requested character limit.
    structured = []
    total = 0
    limit = max(1000, int(max_text_chars))

    prefixes = {
        "h1": "# ",
        "h2": "## ",
        "h3": "### ",
        "h4": "#### ",
        "h5": "##### ",
        "h6": "###### ",
        "li": "- ",
        "blockquote": "> ",
        "fallback": "",
    }

    for item in selected:
        kind = item["kind"]
        value = item["text"]

        if kind == "code":
            line = f"```text\n{value}\n```"
        else:
            line = prefixes.get(kind, "") + value

        remaining = limit - total
        if remaining <= 0:
            break

        if len(line) > remaining:
            # Avoid truncating a block to an unreadably short fragment.
            if remaining >= 200:
                structured.append(line[:remaining])
            break

        structured.append(line)
        total += len(line) + 1

    text = "\n\n".join(structured)
    print(
        f"[EXTRACT DEBUG] URL={page_url} candidates={len(candidates)} "
        f"structured_chars={len(text)} body_chars={len(fallback_text)}"
    )

    # Extract image metadata from the main content area.
    images = []
    seen_images = set()

    for img in main.find_all("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-lazy-src")
            or img.get("data-original")
        )

        if not src:
            continue

        image_url = urljoin(page_url, src)
        parsed_img = urlparse(image_url)

        if (
            parsed_img.scheme not in ("http", "https")
            or image_url in seen_images
        ):
            continue

        seen_images.add(image_url)

        alt = " ".join((img.get("alt") or "").split())[:300]
        image_title = " ".join((img.get("title") or "").split())[:200]

        parent_text = (
            " ".join(img.parent.get_text(" ", strip=True).split())[:500]
            if img.parent else ""
        )

        images.append({
            "url": image_url,
            "alt": alt,
            "title": image_title,
            "width": str(img.get("width", "")),
            "height": str(img.get("height", "")),
            "context": parent_text,
            "relevance": 0.0,
        })

    image_texts = [
        f"{img['alt']} {img['title']} {img['context']}"
        for img in images
    ]

    image_scores = _relevance_scores(image_texts, question)

    for img, score in zip(images, image_scores):
        try:
            area = int(img["width"] or 0) * int(img["height"] or 0)
        except ValueError:
            area = 0

        img["relevance"] = (
            float(score)
            + (0.03 if img["alt"] else 0)
            + (0.03 if area >= 40000 else 0)
        )

    images.sort(key=lambda item: item["relevance"], reverse=True)

    if _same_host(canonical_url, urlparse(page_url).hostname):
        canonical_url = _normalise(canonical_url)
    else:
        canonical_url = page_url

    return title, text, images, canonical_url


def _link_relevance(anchor, link_url, question):
    """Score an internal link using its visible text, title and URL."""
    if not question or not question.strip():
        return 0.0

    anchor_text = " ".join(anchor.get_text(" ", strip=True).split())
    title = " ".join((anchor.get("title") or "").split())
    path_text = urlparse(link_url).path.replace("/", " ").replace("-", " ").replace("_", " ")
    combined = " ".join(part for part in (anchor_text, title, path_text) if part)
    return _relevance_scores([combined], question)[0] if combined.strip() else 0.0


def _rank_pages(pages, question):
    """Rank extracted pages by relevance to the research question."""
    if not pages:
        return pages

    scores = _relevance_scores(
        [
            page.get("title", "") + " " + page.get("text", "")[:18000]
            for page in pages
        ],
        question,
    )

    for page, score in zip(pages, scores):
        page["relevance"] = float(score)

    pages.sort(key=lambda page: page["relevance"], reverse=True)
    return pages


def crawl_website(start_url, question="", max_pages=5, max_text_chars=18000):
    """
    Crawl accessible same-host pages and return structured, ranked results.
    """
    start_url = _public_http_url(start_url)
    base_host = urlparse(start_url).hostname
    requested_url = _normalise(start_url)
    requested_parts = urlparse(requested_url)
    chart_focused = any(term in (question or "").lower() for term in (
        "chart", "graph", "plot", "visualization", "visualisation",
        "axis", "axes", "trend", "readable values", "data points",
        "annual co₂", "co2 emissions", "emissions chart",
    ))

    def is_requested_page(candidate_url):
        try:
            candidate = urlparse(_normalise(candidate_url))
            return (
                (candidate.hostname or "").lower() == (requested_parts.hostname or "").lower()
                and (candidate.port or (443 if candidate.scheme == "https" else 80))
                    == (requested_parts.port or (443 if requested_parts.scheme == "https" else 80))
                and candidate.path.rstrip("/") == requested_parts.path.rstrip("/")
            )
        except Exception:
            return False

    queue = deque([requested_url])
    seen_urls = set()
    seen_canonical_urls = set()
    seen_content = set()
    pages = []

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    page_limit = max(1, min(int(max_pages), 20))

    while queue and len(pages) < page_limit:
        current = _normalise(queue.popleft())

        if current in seen_urls:
            continue

        seen_urls.add(current)

        try:
            safe_url = _public_http_url(current)

            if not _same_host(safe_url, base_host):
                continue

            if not _robots_allowed(safe_url):
                continue

            # Redirects are disabled so a request cannot silently redirect
            # to a different host or private address.
            response = session.get(
                safe_url,
                timeout=TIMEOUT,
                allow_redirects=False,
                stream=True,
            )
            print(
                f"[CRAWLER] URL={safe_url} "
                f"STATUS={response.status_code} "
                f"CONTENT_TYPE={response.headers.get('Content-Type', '')}"
            )

            try:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if location:
                        redirected = urljoin(safe_url, location)
                        if _same_host(redirected, base_host):
                            normal = _normalise(redirected)
                            if normal not in seen_urls and normal not in queue:
                                queue.append(normal)
                    continue

                if response.status_code != 200:
                    continue

                content_type = response.headers.get(
                    "Content-Type", ""
                ).lower()

                if not any(kind in content_type for kind in (
                    "text/html", "application/xhtml+xml"
                )):
                    continue

                chunks = []
                total = 0

                for chunk in response.iter_content(65536):
                    if not chunk:
                        continue

                    total += len(chunk)

                    if total > MAX_RESPONSE_BYTES:
                        break

                    chunks.append(chunk)

                if total > MAX_RESPONSE_BYTES:
                    continue

                raw_html = b"".join(chunks)
                encoding = response.encoding or "utf-8"
                try:
                    html = raw_html.decode(encoding, errors="replace")
                except LookupError:
                    html = raw_html.decode("utf-8", errors="replace")

                final_url = safe_url

            finally:
                response.close()

            title, text, images, canonical_url = _clean_page(
                html,
                final_url,
                question,
                max_text_chars,
            )
            print(
                f"[CRAWLER] EXTRACTED title={title!r} "
                f"characters={len(text)} images={len(images)}"
            )

            if not text.strip():
                continue

            if canonical_url in seen_canonical_urls:
                continue

            seen_canonical_urls.add(canonical_url)

            content_hash = hashlib.sha256(
                " ".join(text.lower().split()).encode("utf-8")
            ).hexdigest()

            if content_hash in seen_content:
                continue

            seen_content.add(content_hash)

            # Provenance must be based on the URL actually fetched, not the page's
            # canonical tag. A related page can declare the requested chart as its
            # canonical URL; that must not make its text/images chart evidence.
            is_target = is_requested_page(final_url)

            # For chart-focused research, related-page images must never be mixed
            # into the requested chart's visual evidence. Related text is retained
            # for general crawling, with provenance marked for the caller.
            if chart_focused and not is_target:
                images = []

            pages.append({
                # Preserve the fetched URL for trustworthy source attribution.
                # Canonical URL is retained separately for diagnostics/deduplication.
                "url": final_url if is_target else canonical_url,
                "canonical_url": canonical_url,
                "title": title,
                "text": text,
                "images": images,
                "relevance": 0.0,
                "content_hash": content_hash,
                "image_count": len(images),
                "is_requested_page": bool(is_target),
                "image_evidence_scope": "requested_page_only" if chart_focused else "crawled_page",
            })

            # Discover internal links. For a focused question, prioritize
            # links whose text/title/path best match the question.
            soup = BeautifulSoup(html, "html.parser")
            discovered_links = []
            discovered_seen = set()

            for anchor in soup.find_all("a", href=True):
                link = urljoin(final_url, anchor["href"])

                if urlparse(link).scheme not in ("http", "https"):
                    continue

                if not _same_host(link, base_host):
                    continue

                normal = _normalise(link)
                if normal == _normalise(final_url) or normal in discovered_seen:
                    continue
                if normal in seen_urls or normal in queue:
                    continue

                discovered_seen.add(normal)
                score = _link_relevance(anchor, normal, question)
                discovered_links.append((score, normal))

            discovered_links.sort(key=lambda item: item[0], reverse=True)
            for _, normal in discovered_links[:MAX_LINKS_PER_PAGE]:
                queue.append(normal)

            time.sleep(0.2)

        except (requests.RequestException, ValueError, OSError) as exc:
            print(f"[CRAWLER ERROR] URL={current} ERROR={exc}")
            print(f"\n[CRAWLER ERROR] URL={current}")
            print(f"[ERROR TYPE] {type(exc).__name__}")
            print(f"[ERROR MESSAGE] {exc}")
            traceback.print_exc()
            continue
        
    return _rank_pages(pages, question)
