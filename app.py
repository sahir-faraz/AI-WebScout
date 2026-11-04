import traceback
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.crawler import crawl_website
from src.extractor import extract_information
from src.rag_pipeline import retrieve_relevant_context
from src.image_extractor import analyze_image
from src.evidence_detector import review_evidence
from src.owid_data import fetch_owid_chart

# Load .env from the same folder as app.py
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

st.set_page_config(page_title="AI-Web Crawler", layout="wide")

# ISO 3166-1 alpha-3 codes. OWID uses other codes for aggregate series such as
# World, Asia, OECD and income groups; those must not appear in country rankings.
ISO3_COUNTRY_CODES = set("""
ABW AFG AGO AIA ALA ALB AND ARE ARG ARM ASM ATA ATF ATG AUS AUT AZE
BDI BEL BEN BES BFA BGD BGR BHR BHS BIH BLM BLR BLZ BMU BOL BRA BRB
BRN BTN BVT BWA CAF CAN CCK CHE CHL CHN CIV CMR COD COG COK COL COM
CPV CRI CUB CUW CXR CYM CYP CZE DEU DJI DMA DNK DOM DZA ECU EGY ERI
ESH ESP EST ETH FIN FJI FLK FRA FRO FSM GAB GBR GEO GGY GHA GIB GIN
GLP GMB GNB GNQ GRC GRD GRL GTM GUF GUM GUY HKG HMD HND HRV HTI HUN
IDN IMN IND IOT IRL IRN IRQ ISL ISR ITA JAM JEY JOR JPN KAZ KEN KGZ
KHM KIR KNA KOR KWT LAO LBN LBR LBY LCA LIE LKA LSO LTU LUX LVA MAC
MAF MAR MCO MDA MDG MDV MEX MHL MKD MLI MLT MMR MNE MNG MNP MOZ MRT
MSR MTQ MUS MWI MYS MYT NAM NCL NER NFK NGA NIC NIU NLD NOR NPL NRU
NZL OMN PAK PAN PCN PER PHL PLW PNG POL PRI PRK PRT PRY PSE PYF QAT
REU ROU RUS RWA SAU SDN SEN SGP SGS SHN SJM SLB SLE SLV SMR SOM SPM
SRB SSD STP SUR SVK SVN SWE SWZ SXM SYC SYR TCA TCD TGO THA TJK TKL
TKM TLS TON TTO TUN TUR TUV TWN TZA UGA UKR UMI URY USA UZB VAT VCT
VEN VGB VIR VNM VUT WLF WSM YEM ZAF ZMB ZWE
""".split())

st.title("AI-Web Crawler")
st.caption("Research public websites using webpage text and image understanding.")

with st.sidebar:
    st.header("Research settings")
    max_pages = st.slider("Maximum pages", 1, 20, 5)
    max_images = st.slider("Maximum images to analyze", 0, 10, 3)
    max_chars = st.slider("Maximum characters per page", 3000, 18000, 9000, step=1000)
    model = st.text_input("Gemini model", value="gemini-3.8-flash")
    st.caption("Images are optional. Keep the limit low to control API usage.")
    enable_rag = st.checkbox("Enable RAG (Transformers + LangChain)", value=True, help="Builds a local vector index of crawled text and retrieves passages relevant to your research question. First use may download the embedding model.")
    rag_top_k = st.slider("RAG passages to retrieve", 1, 10, 5, disabled=not enable_rag)
    enable_evidence_review = st.checkbox("Run Evidence Detective + text–visual comparison", value=True)
    max_claims = st.slider("Maximum claims to review", 1, 15, 8)

url = st.text_input("Website URL", placeholder="https://example.com")
question = st.text_area(
    "Research question (optional for Complete Website Insights)",
    placeholder="Example: What are the main Python data types, and how are they used?",
    height=90,
)

mode = st.radio(
    "Choose research mode",
    ["Research Question", "Complete Website Insights"],
    horizontal=True,
    help="Research Question focuses on your question. Complete Website Insights summarizes the main information found across accessible pages.",
)

# Persist the latest completed report across Streamlit reruns.
if "ai_webscout_last_report" not in st.session_state:
    st.session_state["ai_webscout_last_report"] = ""
if "ai_webscout_last_report_name" not in st.session_state:
    st.session_state["ai_webscout_last_report_name"] = "ai_webscout_report.md"

_research_ran_this_run = False

button_label = (
    "Research Question"
    if mode == "Research Question"
    else "Complete Website Insights"
)

if st.button(button_label, type="primary", disabled=not url.strip()):
    _research_ran_this_run = True
    if mode == "Research Question" and not question.strip():
        st.warning("Enter a research question, or choose Complete Website Insights.")
        st.stop()
    gemini_available = bool(os.getenv("GEMINI_API_KEY"))
    if not gemini_available:
        st.warning(
            "GEMINI_API_KEY is missing. AI analysis will be skipped where possible; "
            "AI-Web Cralwer will still try to crawl pages and report extracted content."
        )
    with st.spinner("Crawling accessible pages..."):
        try:
            crawl_question = question.strip() if mode == "Research Question" else ""
            pages = crawl_website(
                url.strip(),
                crawl_question,
                max_pages=max_pages,
                max_text_chars=max_chars,
            )
        except Exception as exc:
            st.error(f"Could not crawl this website: {exc}")
            st.code(traceback.format_exc())
            st.stop()
    if not pages:
        st.warning("No accessible pages were found. The website may block automated access or disallow crawling.")
        st.stop()

    # For OWID Grapher requests, only the exact requested chart page may support
    # the chart-specific text/image analysis. Related pages remain visible in the
    # crawl table, but cannot be silently presented as evidence for this chart.
    submitted_url = url.strip()
    is_owid_grapher = "ourworldindata.org/grapher/" in submitted_url.lower()
    analysis_pages = pages
    target_page_found = True
    if is_owid_grapher:
        from urllib.parse import urlparse, urldefrag

        def _page_key(value):
            try:
                parsed = urlparse(urldefrag((value or "").strip())[0])
                return ((parsed.hostname or "").lower(), (parsed.path or "/").rstrip("/"))
            except Exception:
                return ("", "")

        requested_key = _page_key(submitted_url)
        analysis_pages = [
            page for page in pages
            if page.get("is_requested_page") is True
            or _page_key(page.get("url", "")) == requested_key
        ]
        target_page_found = bool(analysis_pages)
        if not target_page_found:
            st.warning(
                "The crawler did not retrieve the exact requested chart page. "
                "Related-page text and images will not be used as evidence for this chart; "
                "the direct CSV/metadata source may still be available below."
            )

    # Retrieve structured CSV + metadata for supported Our World in Data Grapher charts.
    # This evidence is kept separate from crawler text and image findings so its provenance stays clear.
    owid_df = None
    owid_metadata = None
    owid_error = None
    ranking_error = None  # Keep report generation safe when no OWID ranking is attempted.
    owid_evidence_markdown = ""
    owid_evidence_prompt = ""
    if is_owid_grapher:
        with st.spinner("Retrieving Our World in Data chart CSV and metadata..."):
            try:
                owid_df, owid_metadata, owid_error = fetch_owid_chart(submitted_url)
            except Exception as exc:
                owid_error = str(exc)

        if owid_error:
            st.warning(f"Could not retrieve structured chart data: {owid_error}")
        elif owid_df is not None and owid_metadata is not None:
            chart_meta = owid_metadata.get("chart", {})
            column_meta = owid_metadata.get("columns", {})
            st.subheader("Structured chart data — CSV source")
            st.caption("Retrieved directly from the chart's CSV and metadata endpoints; shown separately from AI-generated text and image interpretations.")
            st.write(f"**Chart:** {chart_meta.get('title', 'Title not provided')}")
            if chart_meta.get("subtitle"):
                st.write(f"**Description:** {chart_meta['subtitle']}")
            st.write(f"**Source:** {chart_meta.get('citation', 'Citation not provided')}")
            st.write(f"**Rows:** {len(owid_df):,} · **CSV columns:** {', '.join(map(str, owid_df.columns))}")
            year_min = int(owid_df["Year"].min()) if "Year" in owid_df and not owid_df["Year"].isna().all() else None
            year_max = int(owid_df["Year"].max()) if "Year" in owid_df and not owid_df["Year"].isna().all() else None
            if year_min is not None and year_max is not None:
                st.write(f"**Years in CSV:** {year_min}–{year_max}")
            for col_name, details in column_meta.items():
                if col_name in owid_df.columns:
                    unit = details.get("unit") or details.get("shortUnit") or "Unit not specified"
                    description = details.get("descriptionShort", "")
                    st.write(f"**Measure:** {col_name} · **Unit:** {unit}")
                    if description:
                        st.caption(description)
            with st.expander("Preview source CSV", expanded=False):
                st.dataframe(owid_df.head(50), use_container_width=True, hide_index=True)

            # Compute a country-only ranking from the full downloaded CSV.
            # The Code allowlist excludes OWID aggregate series (World, Asia, OECD, etc.).
            value_columns = [c for c in owid_df.columns if c not in {"Entity", "Code", "Year", "Day"}]
            latest_year = year_max
            ranking_markdown = ""
            ranking_error = None
            if latest_year is not None and {"Entity", "Code", "Year"}.issubset(owid_df.columns) and value_columns:
                ranking_value_col = value_columns[0]
                country_rows = owid_df[owid_df["Code"].astype("string").str.upper().isin(ISO3_COUNTRY_CODES)].copy()
                latest_rows = country_rows.loc[country_rows["Year"] == latest_year, ["Entity", "Code", ranking_value_col]].copy()
                latest_rows[ranking_value_col] = pd.to_numeric(latest_rows[ranking_value_col], errors="coerce")
                latest_rows = latest_rows.dropna(subset=[ranking_value_col])
                latest_rows = latest_rows.sort_values(ranking_value_col, ascending=False).drop_duplicates("Code").head(10)
                if not latest_rows.empty:
                    latest_rows = latest_rows.rename(columns={ranking_value_col: "Emissions"})
                    display_rows = latest_rows[["Entity", "Emissions"]].copy()
                    display_rows.insert(0, "Rank", range(1, len(display_rows) + 1))
                    st.subheader(f"Top 10 countries by annual emissions ({latest_year})")
                    st.caption("Calculated from the full CSV, filtered to ISO 3166-1 alpha-3 country codes; regional and income-group aggregates are excluded.")
                    st.dataframe(display_rows, use_container_width=True, hide_index=True)
                    ranking_markdown = (
                        f"### Top countries by annual emissions ({latest_year}; country-code filtered full CSV)\n\n"
                        "| Rank | Country | Emissions (tonnes) |\n|---:|---|---:|\n"
                        + "\n".join(
                            f"| {rank} | {row['Entity']} | {row['Emissions']:,.0f} |"
                            for rank, (_, row) in enumerate(latest_rows.iterrows(), start=1)
                        )
                    )
                else:
                    ranking_error = f"No rows with recognized ISO alpha-3 country codes were available for {latest_year}."
                    st.warning(ranking_error)
            elif owid_df is not None:
                ranking_error = "The CSV is missing the Entity, Code, or Year column required for a country-only ranking."
                st.warning(ranking_error)

            # Create a compact, auditable data note for the downloadable report.
            owid_lines = [
                "## Structured chart data (direct CSV and metadata)",
                "",
                "> These details were retrieved from the chart's CSV/metadata endpoints, not inferred from an image. This section is separate from the AI-generated webpage and image analyses.",
                "",
                f"- Chart: {chart_meta.get('title', 'Not provided')}",
                f"- Description: {chart_meta.get('subtitle', 'Not provided')}",
                f"- Source/citation: {chart_meta.get('citation', 'Not provided')}",
                f"- CSV source URL: {submitted_url.rstrip('/')}.csv",
                f"- Rows retrieved: {len(owid_df):,}",
                f"- Year range in CSV: {year_min if year_min is not None else 'Unavailable'}–{year_max if year_max is not None else 'Unavailable'}",
                "",
                "### Measure metadata",
                "",
            ]
            for col_name, details in column_meta.items():
                if col_name in owid_df.columns:
                    owid_lines.extend([
                        f"- **{col_name}** — unit: {details.get('unit') or details.get('shortUnit') or 'not specified'}; "
                        f"description: {details.get('descriptionShort', 'not provided')}; "
                        f"timespan: {details.get('timespan', 'not provided')}"
                    ])
            # Include a compact recent-year sample for the countries selected in metadata, when present.
            selected = chart_meta.get("selection", [])
            value_columns = [c for c in owid_df.columns if c not in {"Entity", "Code", "Year", "Day"}]
            if selected and "Entity" in owid_df.columns and "Year" in owid_df.columns and value_columns:
                sample = owid_df[owid_df["Entity"].isin(selected)].copy()
                if not sample.empty:
                    sample = sample.sort_values("Year").groupby("Entity", group_keys=False).tail(3)
                    sample = sample[[c for c in ["Entity", "Year"] + value_columns if c in sample.columns]]
                    owid_lines.extend(["", "### Recent CSV rows for chart-selected entities", "", "```csv\n" + sample.to_csv(index=False).strip() + "\n```"])
            if ranking_markdown:
                owid_lines.extend(["", ranking_markdown])
            owid_evidence_markdown = "\n".join(owid_lines)
            # Bounded, source-labelled dataset evidence is sent to both Gemini workflows.
            prompt_lines = [
                f"CSV source URL: {submitted_url.rstrip('/')}.csv",
                f"Chart title: {chart_meta.get('title', 'Not provided')}",
                f"Chart description: {chart_meta.get('subtitle', 'Not provided')}",
                f"Source citation: {chart_meta.get('citation', 'Not provided')}",
                f"Full dataset year range from all downloaded CSV rows: {year_min}–{year_max}",
                "Important: any recent-year rows included below are only a sample for readability; they do NOT define the full dataset time range.",
            ]
            for col_name, details in column_meta.items():
                if col_name in owid_df.columns:
                    prompt_lines.append(f"Metric: {col_name}; unit: {details.get('unit') or details.get('shortUnit') or 'not specified'}; description: {details.get('descriptionShort', 'not provided')}")
            if selected and "Entity" in owid_df.columns and "Year" in owid_df.columns and value_columns:
                prompt_sample = owid_df[owid_df["Entity"].isin(selected)].sort_values("Year").groupby("Entity", group_keys=False).tail(3)
            else:
                prompt_sample = owid_df.tail(15)
            prompt_lines.extend(["Actual downloaded CSV rows (recent sample only):", prompt_sample.to_csv(index=False)])
            if ranking_markdown:
                prompt_lines.extend([
                    "Verified all-country ranking computed from the full CSV for the latest year:",
                    ranking_markdown,
                    "Use this ranking when discussing the highest-emitting countries. Do not describe the recent sample as the full dataset."
                ])
            owid_evidence_prompt = "\n".join(prompt_lines)[:12000]

    st.subheader("Crawled pages")
    st.dataframe(pd.DataFrame([{
        "Title": p.get("title", ""), "URL": p.get("url", ""),
        "Characters": len(p.get("text", "")), "Images found": len(p.get("images", [])), "Relevance": round(float(p.get("relevance", 0)), 3)
    } for p in pages]), use_container_width=True, hide_index=True)

    retrieval_context = ""
    rag_error = None
    if enable_rag and question.strip():
        with st.spinner("Preparing Transformer embeddings and retrieving relevant passages..."):
            try:
                retrieval_context = retrieve_relevant_context(
                    analysis_pages, question.strip(), top_k=rag_top_k
                )
                if retrieval_context:
                    st.success("RAG retrieval completed. Relevant source passages will be included in the Gemini prompt.")
                else:
                    st.info("RAG did not return passages; continuing with the existing webpage-text workflow.")
            except Exception as exc:
                rag_error = str(exc)
                st.warning(f"RAG is unavailable, so the original analysis workflow will be used. Details: {rag_error}")

    text_report_error = None
    with st.spinner("Analyzing webpage text with Gemini..."):
        try:
            text_report = extract_information(
                analysis_pages,
                question.strip(),
                model,
                max_chars,
                mode=("complete" if mode == "Complete Website Insights" else "question"),
                structured_evidence=owid_evidence_prompt,
                retrieval_context=retrieval_context,
            )
        except Exception as exc:
            text_report_error = str(exc)
            st.error(f"Text analysis failed: {text_report_error}")
            text_report = ""
    if not text_report or not str(text_report).strip():
        if not text_report_error:
            text_report_error = "The model returned an empty response without raising an exception."
        # Local fallback: keep the product useful when Gemini is unavailable or quota-limited.
        local_sections = []
        for page in analysis_pages:
            page_text = (page.get("text") or "").strip()
            if page_text:
                local_sections.append(
                    f"### {page.get('title') or page.get('url') or 'Retrieved page'}\\n\\n"
                    f"Source: {page.get('url', '')}\\n\\n"
                    f"{page_text[:5000]}"
                )
        if local_sections:
            text_report = (
                "## Extracted webpage content (local fallback; not AI-generated)\\n\\n"
                "AI analysis was unavailable, so the following is a source-labelled extract of "
                "retrieved page content. It is not an AI summary or independent fact-check.\\n\\n"
                + "\\n\\n---\\n\\n".join(local_sections)
            )
            st.info("Showing extracted webpage content as a local fallback because AI analysis was unavailable.")
        else:
            st.warning(
                "AI text analysis did not return a report and no readable page text was extracted. "
                "The website may require JavaScript, login, or access that the crawler cannot use."
            )

    image_results = []
    if max_images:
        candidates, seen = [], set()
        # Prefer visuals likely to contain data or meaningful text. This metadata
        # prefilter avoids sending obvious logos/icons/decorative images to Gemini.
        positive_terms = (
            "chart", "graph", "plot", "diagram", "infographic", "data visual",
            "map", "table", "timeline", "figure", "illustration", "schematic",
            "statistics", "statistic", "trend", "comparison", "report", "results",
            "mission", "timeline", "label", "caption", "text"
        )
        negative_terms = (
            "logo", "icon", "avatar", "badge", "social", "facebook", "twitter",
            "instagram", "youtube", "pixel", "tracking", "sprite", "thumbnail",
            "favicon", "banner", "background", "separator", "button"
        )

        for page in analysis_pages:
            for img in page.get("images", []):
                img_url = img.get("url")
                if not img_url or img_url in seen:
                    continue
                seen.add(img_url)
                alt = img.get("alt", "") or ""
                context = img.get("context", "") or ""
                page_url = img.get("source_page_url", page.get("url", ""))
                haystack = f"{img_url} {alt} {context} {page.get('title', '')}".lower()
                positive_hits = sum(1 for term in positive_terms if term in haystack)
                negative_hits = sum(1 for term in negative_terms if term in haystack)
                # Require positive signals and reject strong decorative signals.
                # Page relevance is retained as a tie-breaker, not as proof of content.
                if positive_hits == 0 or negative_hits >= 2:
                    continue
                candidates.append({
                    "url": img_url, "alt": alt,
                    "page_url": page_url,
                    "relevance": float(img.get("relevance", 0.0)),
                    "context": context,
                    "visual_score": positive_hits - (2 * negative_hits),
                })

        candidates.sort(
            key=lambda item: (item["visual_score"], item["relevance"]),
            reverse=True,
        )
        if candidates:
            st.subheader("Image analysis")
            count = min(max_images, len(candidates))
            st.caption(
                f"Found {len(candidates)} likely chart, diagram, map, infographic, or text-rich "
                f"image candidates after metadata filtering. Analyzing up to {count}. "
                "This filter uses image URLs, alt text, page titles, and nearby context; "
                "some relevant images without descriptive metadata may be missed."
            )
            progress = st.progress(0)
            image_quota_exhausted = False
            for i, candidate in enumerate(candidates[:count]):
                if image_quota_exhausted:
                    break
                try:
                    with st.spinner(f"Analyzing image {i + 1} of {count}..."):
                        result = analyze_image(candidate["url"], candidate["alt"], candidate["page_url"], model)
                    if result:
                        # Preserve the metadata from the exact page that exposed this image.
                        # This helps the evidence reviewer avoid comparing unrelated charts/pages.
                        result["context"] = candidate.get("context", "")
                        result["relevance"] = candidate.get("relevance", 0.0)
                        image_results.append(result)
                except Exception as exc:
                    error_text = str(exc)
                    is_quota_error = (
                        "429" in error_text
                        or "RESOURCE_EXHAUSTED" in error_text.upper()
                        or "quota exceeded" in error_text.lower()
                    )
                    message = (
                        "Image analysis skipped because the Gemini API quota is exhausted. "
                        "Text and structured CSV analysis can still complete. Check your API quota/billing "
                        "or retry after the quota resets."
                        if is_quota_error else f"Image skipped: {error_text}"
                    )
                    image_results.append({"image_url": candidate["url"], "page_url": candidate["page_url"],
                                          "alt": candidate["alt"], "analysis": message})
                    if is_quota_error:
                        image_quota_exhausted = True
                        st.warning(message)
                progress.progress((i + 1) / count)
            if image_quota_exhausted and count > 1:
                st.info("Remaining image-analysis calls were skipped to avoid repeating the quota error.")
        else:
            st.info(
                "No images matched the chart/diagram/map/infographic/text-rich metadata filter. "
                "Relevant visuals without descriptive alt text, filenames, or nearby context may have been missed."
            )

    evidence_report = None
    ai_quota_or_auth_error = bool(
        text_report_error and any(term in text_report_error.lower() for term in
                                  ("429", "resource_exhausted", "quota", "api key", "unauthorized", "401", "403"))
    )
    if enable_evidence_review and gemini_available and not ai_quota_or_auth_error:
        with st.spinner("Checking claims against webpage text and available visual evidence..."):
            try:
                evidence_report = review_evidence(
                    analysis_pages,
                    image_results=image_results,
                    question=question.strip(),
                    model=model,
                    max_claims=max_claims,
                    max_input_chars=min(max_chars * max(1, len(pages)), 18000),
                    structured_evidence=owid_evidence_prompt,
                )
            except Exception as exc:
                st.warning(f"Evidence review could not complete: {exc}")
    elif enable_evidence_review:
        st.info("Evidence Detective was skipped because Gemini is unavailable or a quota/authentication error was detected.")

    if evidence_report:
        st.subheader("Evidence Detective")
        st.caption(evidence_report.get("overall_note", "These labels assess consistency with the supplied website evidence, not absolute truth."))
        claims = evidence_report.get("claims", [])
        if not claims:
            st.info("No checkable claims were returned for review.")
        for idx, claim in enumerate(claims, 1):
            status = claim.get("status", "Unverified")
            with st.expander(f"{idx}. {status}: {claim.get('claim', 'Claim')}", expanded=(idx == 1)):
                st.write(claim.get("reason", "No explanation supplied."))
                st.caption(f"Confidence: {claim.get('confidence', 'Low')} · Human review needed: {'Yes' if claim.get('human_review_needed', True) else 'No'}")
                text_evidence = claim.get("text_evidence", [])
                if text_evidence:
                    st.markdown("**Text evidence**")
                    for evidence in text_evidence:
                        excerpt = evidence.get("excerpt", "")
                        source = evidence.get("source_url", "")
                        if excerpt:
                            st.write(f"> {excerpt}")
                        if source:
                            st.markdown(f"Source: [{source}]({source})")
                data_evidence = claim.get("data_evidence", [])
                if data_evidence:
                    st.markdown("**Structured CSV evidence**")
                    for data_item in data_evidence:
                        if data_item.get("excerpt"):
                            st.write(f"> {data_item['excerpt']}")
                        if data_item.get("source_url"):
                            st.markdown(f"Dataset: [{data_item['source_url']}]({data_item['source_url']})")
                visual_evidence = claim.get("visual_evidence", [])
                if visual_evidence:
                    st.markdown("**Visual evidence**")
                    for evidence in visual_evidence:
                        finding = evidence.get("finding", "")
                        image_url = evidence.get("image_url", "")
                        page_url = evidence.get("page_url", "")
                        if finding:
                            st.write(finding)
                        if image_url:
                            st.markdown(f"Image: [{image_url}]({image_url})")
                        if page_url:
                            st.markdown(f"Page: [{page_url}]({page_url})")

    st.subheader("Research summary — webpage text only")
    st.caption("This section is generated from extracted webpage text only. Image analysis and claim-level comparisons, when available, are reported separately below.")
    if is_owid_grapher and not target_page_found:
        st.info("No text summary was generated from a related page because it did not match the requested chart URL. Use the structured CSV/metadata section for chart facts.")
    else:
        if text_report and str(text_report).strip():
            st.markdown(text_report)
        else:
            st.warning(f"No AI text report was generated. Reason: {text_report_error or 'unknown error'}")
    if image_results:
        st.subheader("Findings from images")
        for idx, result in enumerate(image_results, 1):
            with st.expander(f"Image {idx}: {result.get('alt') or result.get('image_url', 'Image')}", expanded=(idx == 1)):
                st.write(result.get("analysis", "No analysis returned."))
                st.markdown(f"**Image source:** [{result.get('image_url')}]({result.get('image_url')})")
                st.markdown(f"**Found on page:** [{result.get('page_url')}]({result.get('page_url')})")

    report = (
        f"# AI-Web Crawler Research\n\n"
        f"**Website:** {url.strip()}\n\n"
        f"**Mode:** {mode}\n\n"
        f"**Question:** {question.strip() or 'Not specified'}\n\n"
        f"## Webpage text findings (text-only analysis)\n\n"
        f"> Scope: this section is generated from extracted webpage text only. It does not indicate whether image analysis ran. See the separate Image findings and Evidence Detective sections for visual analysis and comparisons.\n\n"
        f"{('The exact requested chart page was not retrieved; related-page text was excluded from chart-specific analysis.' if is_owid_grapher and not target_page_found else (str(text_report).strip() if text_report and str(text_report).strip() else 'No AI text report generated. Reason: ' + (text_report_error or 'The model returned an empty response without an exception.')))}\n"
    )
    if text_report_error:
        report += "\n## AI text analysis status\n\n" + text_report_error + "\n"
    if image_results:
        report += "\n## Image findings\n"
        for i, item in enumerate(image_results, 1):
            report += f"\n### Image {i}\n\n- Image: {item.get('image_url', '')}\n- Page: {item.get('page_url', '')}\n\n{item.get('analysis', '')}\n"
    if owid_evidence_markdown:
        report += "\n" + owid_evidence_markdown + "\n"
    if ranking_error:
        report += "\n## Country ranking status\n\n" + ranking_error + "\n"
    if owid_error and "ourworldindata.org/grapher/" in url.strip().lower():
        report += ("\n## Structured chart data retrieval\n\n"
                   f"CSV/metadata retrieval did not complete: {owid_error}\n")
    if evidence_report:
        report += "\n## Evidence Detective results\n\n"
        report += (evidence_report.get("overall_note", "") + "\n")
        for i, claim in enumerate(evidence_report.get("claims", []), 1):
            report += f"\n### {i}. {claim.get('status', 'Unverified')} — {claim.get('claim', '')}\n\n"
            report += f"{claim.get('reason', '')}\n\n"
            report += f"Confidence: {claim.get('confidence', 'Low')} | Human review needed: {claim.get('human_review_needed', True)}\n"
            for evidence in claim.get("text_evidence", []):
                if isinstance(evidence, dict):
                    report += f"\n- Text evidence: {evidence.get('excerpt', '')} (Source: {evidence.get('source_url', '')})\n"
            for evidence in claim.get("data_evidence", []):
                if isinstance(evidence, dict):
                    report += f"\n- Structured CSV evidence: {evidence.get('excerpt', '')} (Source: {evidence.get('source_url', '')})\n"
            for evidence in claim.get("visual_evidence", []):
                if isinstance(evidence, dict):
                    report += f"\n- Visual evidence: {evidence.get('finding', '')} (Image: {evidence.get('image_url', '')}; Page: {evidence.get('page_url', '')})\n"
    # Save the finished report so it can be restored after Streamlit reruns.
    st.session_state["ai_webscout_last_report"] = report
    st.session_state["ai_webscout_last_report_name"] = "ai_webscout_report.md"

    # Keep one user-facing export. The complete Markdown report includes
    # image findings, structured chart data, and Evidence Detective results.
    st.download_button(
        "Download complete Markdown report",
        report,
        file_name="ai_webscout_report.md",
        mime="text/markdown",
        key="download_complete_report",
    )

# On reruns (for example, after changing tabs/widgets), restore the last report.
# Avoid duplicate output on the run that generated the report above.
if not _research_ran_this_run and st.session_state.get("ai_webscout_last_report"):
    st.subheader("Previous research report")
    st.caption("Restored from this Streamlit session. Run new research to replace it.")
    st.markdown(st.session_state["ai_webscout_last_report"])
    st.download_button(
        "Download complete Markdown report",
        st.session_state["ai_webscout_last_report"],
        file_name=st.session_state.get("ai_webscout_last_report_name", "ai_webscout_report.md"),
        mime="text/markdown",
        key="download_saved_report",
    )

st.divider()
st.caption("Evidence labels describe consistency with the pages retrieved, not absolute truth. Review flagged claims manually. The crawler respects robots.txt and does not bypass logins or paywalls.")
