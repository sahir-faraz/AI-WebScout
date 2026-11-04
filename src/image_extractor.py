"""Download public webpage images and analyze their visual contents with Gemini."""
import ipaddress
import os
import socket
from urllib.parse import urlparse

import requests
from google import genai
from google.genai import types

USER_AGENT = "AI-WebScout/1.1 (image research prototype)"
TIMEOUT = 15
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _validate_public_image_url(url: str) -> str:
    """Accept only public HTTP(S) image URLs and reject private-network hosts."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Image URL must use public HTTP or HTTPS.")
    if parsed.username or parsed.password:
        raise ValueError("Image URLs containing credentials are not supported.")
    try:
        infos = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve image host: {parsed.hostname}") from exc
    addresses = {ipaddress.ip_address(info[4][0]) for info in infos}
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Private or non-public image hosts are not allowed.")
    return url


def _download_image(image_url: str) -> tuple[bytes, str]:
    """Fetch a small image from a public URL without following redirects."""
    safe_url = _validate_public_image_url(image_url)
    with requests.get(
        safe_url,
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
        stream=True,
        allow_redirects=False,
    ) as response:
        if response.status_code in (301, 302, 303, 307, 308):
            raise ValueError("Image URL redirected. Use the final image URL directly.")
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
        if content_type not in allowed_types:
            raise ValueError(
                f"URL did not return a supported image (Content-Type: {content_type or 'unknown'})."
            )
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_IMAGE_BYTES:
            raise ValueError("Image is larger than the 8 MB analysis limit.")
        data = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            data.extend(chunk)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("Image is larger than the 8 MB analysis limit.")
    if not data:
        raise ValueError("The image URL returned an empty file.")
    return bytes(data), content_type


def analyze_image(image_url, alt_text="", page_url="", model="gemini-2.5-flash"):
    """Analyze actual visual contents using Gemini multimodal input."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from your .env file.")
    image_bytes, mime_type = _download_image(image_url)
    client = genai.Client(api_key=api_key)
    prompt = f"""
You are the visual research assistant for AI-WebScout. Inspect the supplied image itself,
not just its filename or alt text.

First classify the image as a chart/map, infographic, diagram, photograph, screenshot, or other visual.

For charts and maps, inspect these separately and report only what is visibly readable:
- Title or heading
- X/Y axes, geographic labels, units, legend, scale, time range, and source note
- Numeric values, country/region names, and trends
- Missing or unreadable elements that prevent interpretation

Rules:
- Treat the image pixels as the primary visual evidence; URL, page title, alt text,
  and surrounding text are provenance/context only.
- If the visible chart title/topic does not match the research question, label it
  "NOT_RELEVANT_TO_REQUESTED_CHART" and do not answer the requested chart question
  from this image.
- Do not infer a chart's metric from its URL, page title, or alt text alone.
- Never invent values, rankings, units, dates, or trends.
- If the title, legend, scale, or units are missing/unreadable, say so explicitly.
- If the chart cannot be interpreted reliably, state: "Insufficient readable visual evidence to interpret this chart."
- Separate direct visual observations from interpretations.
- Do not identify people in photos.

Context:
- Image alt text: {alt_text or "Not provided"}
- Source page URL: {page_url or "Not provided"}
- Image URL: {image_url}

Separate direct visual observations from interpretations. For charts, explain only values and
trends that can be read confidently. Do not identify people in photos.
"""
    response = client.models.generate_content(
        model=model or "gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
    )
    analysis = (response.text or "").strip()
    if not analysis:
        raise RuntimeError("Gemini returned no image-analysis text.")
    return {
        "image_url": image_url,
        "page_url": page_url,
        "alt": alt_text or "",
        "analysis": analysis,
        "evidence_scope": "image_only",
    }
