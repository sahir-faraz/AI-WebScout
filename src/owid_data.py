
import requests
import pandas as pd
from io import StringIO
from urllib.parse import urlparse

HEADERS = {"User-Agent": "AI-WebScout/1.0"}

def fetch_owid_chart(url: str):
    """Fetch CSV data and metadata for an OWID Grapher chart."""

    parsed = urlparse(url)

    if (
        parsed.scheme != "https"
        or parsed.hostname != "ourworldindata.org"
        or not parsed.path.startswith("/grapher/")
    ):
        return None, None, "Not a supported OWID Grapher URL."

    slug = parsed.path.rstrip("/").split("/")[-1]

    if not slug or "." in slug:
        return None, None, "Invalid chart URL."

    base = f"https://ourworldindata.org/grapher/{slug}"

    try:
        csv_response = requests.get(
            f"{base}.csv",
            headers=HEADERS,
            timeout=30,
        )
        csv_response.raise_for_status()

        metadata_response = requests.get(
            f"{base}.metadata.json",
            headers=HEADERS,
            timeout=30,
        )
        metadata_response.raise_for_status()

        df = pd.read_csv(StringIO(csv_response.content.decode("utf-8-sig")))
        metadata = metadata_response.json()

        if df.empty or len(df.columns) < 3:
            return None, None, "Chart data is empty or unexpected."

        if not {"Entity", "Code", "Year"}.issubset(df.columns):
            return None, None, (
                "CSV does not have the expected Entity, Code, "
                "and Year columns."
            )

        return df, metadata, None

    except requests.RequestException as exc:
        return None, None, f"Could not download chart data: {exc}"
    except (ValueError, pd.errors.ParserError) as exc:
        return None, None, f"Could not parse chart data: {exc}"
