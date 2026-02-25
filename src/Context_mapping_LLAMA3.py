"""
Context_mapping_LLAMA3.py — LLaMA3-based contextual address inference engine.

This script:
1. Sends the raw input to a locally-running LLaMA3 model (via Ollama)
   with a structured prompt to extract place, city, district, and state.
2. Fuzzy-matches the returned city against a reference CSV to obtain
   latitude and longitude.
3. Persists every correction to PostgreSQL.

Requires: Ollama running at http://localhost:11434 with the ``llama3`` model.

Usage:
    python Context_mapping_LLAMA3.py "I visited the Taj in Agra last week"
"""

# ── Encoding bootstrap ────────────────────────────────────────────────────
import sys
import io
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    os.environ["PYTHONIOENCODING"] = "utf-8:replace"
    os.environ["PYTHONUNBUFFERED"] = "1"
except Exception as exc:
    print(f"FATAL: encoding setup failed: {exc}", file=sys.__stderr__)

# ── Imports ────────────────────────────────────────────────────────────────
import json
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import psycopg2
import requests
from rapidfuzz import fuzz, process

# ── Paths & configuration ─────────────────────────────────────────────────
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
CSV_PATH: str = os.path.join(SCRIPT_DIR, "final_training_geocoded.csv")
OLLAMA_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "llama3"

# ── Database configuration ─────────────────────────────────────────────────
DB_CONFIG: Dict[str, str] = {
    "dbname": os.getenv("DB_NAME", "Address_corrector"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

# ── LLaMA3 prompt template ────────────────────────────────────────────────
_PROMPT_TEMPLATE = """You are an address correction AI. Extract the most likely corrected place name and its corresponding city mostly in India where this event might have occurred or place where it is famous for. Respond only in this exact format and do not add any other information:

Place: <corrected place>
City: <main city>
district: <district name>
state: <state name>

If nothing found, respond:
Place: NA
City: NA
district: NA
state: NA

Input: {text}
Output:"""

# Regex to parse the structured LLaMA3 response
_RESPONSE_RE = re.compile(
    r"Place:\s*(.*?)\s*City:\s*(.*?)\s*district:\s*(.*?)\s*state:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_custom_data(csv_path: str) -> pd.DataFrame:
    """
    Load the reference CSV and prepare a clean ``correct_address_clean``
    column for fuzzy matching.
    """
    df = pd.read_csv(csv_path, dtype=str)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df["correct_address_clean"] = df["correct_address"].str.lower().str.strip()
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════

def match_city_to_csv(
    city: str, df: pd.DataFrame, threshold: int = 70
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Fuzzy-match *city* against the reference CSV and return
    (matched_address, latitude, longitude) or (None, None, None).
    """
    city_lower = city.lower().strip()
    choices = df["correct_address_clean"].dropna().unique()
    match = process.extractOne(city_lower, choices, scorer=fuzz.WRatio)

    if match and match[1] >= threshold:
        row = df[df["correct_address_clean"] == match[0]].iloc[0]
        return row["correct_address"], row["latitude"], row["longitude"]

    return None, None, None


def correct_location_with_llama3(
    text: str,
) -> Tuple[str, str, str, str]:
    """
    Send *text* to LLaMA3 (Ollama) and parse the structured response.

    Returns (place, city, district, state) – each ``"NA"`` on failure.
    """
    prompt = _PROMPT_TEMPLATE.format(text=text)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp_text = response.json().get("response", "").strip()

        match = _RESPONSE_RE.search(resp_text)
        if match:
            return tuple(match.group(i).strip() for i in range(1, 5))  # type: ignore[return-value]

        print(f"Regex parse failed for response: {resp_text}", file=sys.stderr)

    except Exception as exc:
        print(f"LLaMA3 error: {exc}", file=sys.stderr)

    return "NA", "NA", "NA", "NA"


# ── Database persistence ──────────────────────────────────────────────────

def save_to_db(
    input_text: str, output_text: str, lat: Any, lon: Any
) -> None:
    """Insert a correction record into PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO corrections (input_text, model_used, output_text, latitude, longitude)
               VALUES (%s, %s, %s, %s, %s);""",
            (input_text, "LLaMA", output_text, lat, lon),
        )
        conn.commit()
        cur.close()
        conn.close()
        print("Saved to database.", file=sys.stderr)
    except Exception as exc:
        print(f"Database error: {exc}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Read input, query LLaMA3, fuzzy-match, emit JSON, and persist."""
    # ── Read input ──
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
    else:
        user_input = input("Enter a sentence: ").strip()

    # ── Load reference CSV ──
    df = load_custom_data(CSV_PATH)

    # ── Query LLaMA3 ──
    place, city, district, state = correct_location_with_llama3(user_input)

    # ── Fuzzy-match city for coordinates ──
    lat: Any = "NA"
    lon: Any = "NA"

    if city.upper() != "NA":
        matched_city, matched_lat, matched_lon = match_city_to_csv(city, df)
        if matched_lat is not None:
            lat = float(matched_lat)
        if matched_lon is not None:
            lon = float(matched_lon)

    # ── Output JSON ──
    result = {
        "corrected_address": city,
        "latitude": lat,
        "longitude": lon,
        "district": district,
        "state": state,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # ── Persist ──
    save_to_db(
        user_input, city,
        None if lat == "NA" else lat,
        None if lon == "NA" else lon,
    )


if __name__ == "__main__":
    main()
