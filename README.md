# AI-WebScout — Evidence Detective + Text–Visual Mismatch Detection

AI-WebScout researches accessible public webpages and produces a source-linked report. It combines:

1. **Respectful crawling** — follows same-host links, checks `robots.txt`, removes common page boilerplate, deduplicates and ranks content.
2. **Website research** — Gemini generates a question-focused answer or broader insights from the pages actually retrieved.
3. **Image understanding** — downloads eligible public images and asks Gemini to describe visible content, including readable chart labels when available.
4. **Evidence Detective** — identifies checkable claims and compares them with extracted webpage evidence. Statuses: `Supported`, `Partially supported`, `Contradicted`, `Unverified`, and `Potential text-visual mismatch`.
5. **Reviewable outputs** — Streamlit results, a complete Markdown report, image-findings CSV, and claim-review CSV.

**Important limitation:** this is evidence triage, not a fact oracle. Gemini can misread pages, images, charts, or its own evidence. A status measures consistency with the retrieved material, not absolute truth. All potential text–visual mismatches and high-impact claims should be reviewed by a person. Numerical chart claims should only be checked when values/labels are actually legible. JavaScript-heavy sites, blocked pages, PDFs, inaccessible images, and pages outside the crawl limit may not be included.

## Project structure

```text
AI-WebScout-Final/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── __init__.py
│   ├── crawler.py
│   ├── extractor.py
│   ├── image_extractor.py
│   └── evidence_detector.py
└── tests/
    ├── test_crawler_helpers.py
    └── test_evidence_detector.py
```

## Install on Windows / VS Code

Run PowerShell commands from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace `your_gemini_api_key_here` with your own key from Google AI Studio. Never commit `.env` or paste your key into chat. The default model field is `gemini-2.5-flash`; if your API account does not have access to it, choose a model that is available in your account in the sidebar.

## Run

```powershell
streamlit run app.py
```

Enter a public website URL. Choose **Research Question** and enter a specific question, or choose **Complete Website Insights**. Keep image count at 2–3 for an initial run to control API usage. The Evidence Detective option is enabled by default.

## Test

First run offline unit tests and compile checks:

```powershell
python -m py_compile app.py src\crawler.py src\extractor.py src\image_extractor.py src\evidence_detector.py
pytest -q
```

Then run the app with a site you are allowed to access. Validate these cases:

- **Supported:** page says “The report covers 100 participants”; ask whether it covers 100 participants.
- **Partially supported:** page says “60% of respondents improved”; ask whether everyone improved.
- **Contradicted:** page says sales fell from January to March; test a claim that sales rose in that period.
- **Unverified:** ask whether a product is “the fastest in the industry” when the retrieved pages provide no benchmark.
- **Potential text–visual mismatch:** use a page with a clearly legible chart and written statement describing the same metric and time range, but make the direction conflict. Ensure the relevant image is among the images analyzed.

Do not evaluate the feature from a single run. Make a small gold set of at least 20 claims (5 per main evidence class), label them manually, and track status accuracy, fabricated/misquoted excerpts, source URL correctness, and false mismatch alerts. Keep chart mismatch tests separate and record whether a human confirmed each alert.

## Troubleshooting

- **Missing API key:** check that `.env` is in the same folder as `app.py` and contains `GEMINI_API_KEY=...`; restart Streamlit after editing.
- **Model not found / access denied:** select a supported Gemini model in the sidebar.
- **No pages found:** test a public HTML page, confirm network access, and respect the site's robots rules. Some sites block automated clients or require JavaScript rendering.
- **Image skipped:** some hosts block direct downloads or use redirects. The app should continue with the remaining evidence; it must not treat a skipped image as a visual finding.
- **API quota or network errors:** lower maximum pages, maximum images, and claims per run.

## Safety and scope

Only analyze public pages you are authorized to access. This app does not log in, bypass paywalls, or intentionally bypass `robots.txt`. Do not use its generated labels as sole support for medical, legal, financial, safety-critical, or other high-impact decisions.

## RAG upgrade: Transformers + LangChain

The updated project adds `src/rag_pipeline.py` and an **Enable RAG (Transformers + LangChain)** option in the Streamlit sidebar.

Workflow when RAG is enabled and a research question is supplied:

1. The existing crawler retrieves accessible page text.
2. LangChain's `RecursiveCharacterTextSplitter` splits the text into overlapping passages.
3. `sentence-transformers/all-MiniLM-L6-v2` (a pretrained Transformer model) converts passages into semantic vectors.
4. FAISS builds an in-memory vector index and retrieves the passages most similar to the user's question.
5. The retrieved passages, with their original page titles and URLs, are added to the Gemini prompt. Gemini writes the final answer.

**Important:** Transformers provide the embedding model; LangChain orchestrates splitting/retrieval; FAISS is the local vector index; Gemini still generates the answer. The RAG index is currently created in memory for each research run and is not persisted. The first use may download the embedding model and can take longer. If RAG dependencies/model download fail, the app displays a warning and falls back to the existing workflow. For Complete Website Insights without a question, RAG retrieval is skipped.

Install the new dependencies from the project folder:

```powershell
python -m pip install -r requirements.txt
```

A research question is recommended to get the most useful retrieval results. Review cited source URLs and retrieved passages; RAG improves relevance but does not guarantee correctness.
