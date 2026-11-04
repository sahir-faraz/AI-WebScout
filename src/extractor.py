
"""Generate website research summaries using the Gemini API."""

import os
from google import genai

DEFAULT_MODEL = "gemini-3.8-flash"


def extract_information(pages, question="", model=DEFAULT_MODEL, max_chars=9000, mode="question", structured_evidence="", retrieval_context=""):
    """Summarize crawled webpage text using Gemini."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from your .env file.")

    if not pages:
        return "No webpage content was available to analyze."

    sections = []
    remaining = max(1000, int(max_chars))

    for page in pages:
        if remaining <= 0:
            break

        title = page.get("title") or "Untitled page"
        url = page.get("url") or ""
        content = (page.get("text") or "").strip()

        if not content:
            continue

        section = f"Title: {title}\nURL: {url}\nContent:\n{content}"
        section = section[:remaining]
        sections.append(section)
        remaining -= len(section)

    if not sections:
        return "The crawled pages did not contain readable text."

    prompt = f"""
You are the research assistant for AI-WebScout.

Answer the user's question using the supplied evidence below. Do not invent facts.
Clearly identify missing information and distinguish documented facts from interpretations.
If structured CSV evidence is supplied, treat its numeric rows and metadata as the primary
source for numerical chart claims. Do not replace CSV values with visual estimates. If a value
is not in the supplied rows, say it was not verified. Keep webpage text and dataset evidence distinct.
Use headings and concise bullet points. Include source URLs where relevant.

Research mode: {mode}
User's question:
{question.strip() or "Provide a concise summary of the website."}

STRUCTURED DATASET EVIDENCE (CSV + metadata):
{structured_evidence or "No structured dataset evidence was available for this URL."}

RAG RETRIEVED PASSAGES (Transformer embeddings + LangChain retrieval; may be empty):
{retrieval_context or "RAG retrieval was not enabled or returned no passages."}

Treat retrieved passages as pointers to relevant source text, not as a replacement for the original source. Preserve the supplied source URLs. If passages conflict, disclose the conflict.

Webpage content:
{chr(10).join(sections)}
"""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=prompt,
    )

    result = response.text
    if not result or not result.strip():
        raise RuntimeError("Gemini returned an empty research summary.")

    return result.strip()
