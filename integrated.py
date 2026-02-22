"""
integrated.py — Central routing engine for the Indian Address Correction System.

This module decides how to process a piece of raw/noisy address text:

1. **Direct geocode** – if the input exactly matches a known city that
   appears in only one Indian state.
2. **T5 model** – for short inputs, known abbreviations, or phrases
   containing prepositions (e.g. "near blr").
3. **LLaMA3 model** – for contextual/long-form inputs, or when the
   matched city is ambiguous (exists in multiple states).

The script can be run standalone for testing or invoked as a subprocess
by the FastAPI backend.
"""

# ── Encoding bootstrap (must run before any I/O) ──────────────────────────
import sys
import io
import os

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

# ── Standard-library & third-party imports ─────────────────────────────────
import datetime
import json
import logging
import re
import subprocess
from typing import Dict, Optional, Set, Tuple

import pandas as pd
import spacy

# ── Constants & paths ──────────────────────────────────────────────────────
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
T5_SCRIPT_PATH: str = os.path.join(SCRIPT_DIR, "T5_fine_tuned.py")
LLAMA3_SCRIPT_PATH: str = os.path.join(SCRIPT_DIR, "Context_mapping_LLAMA3.py")
LOG_FILE: str = os.path.join(SCRIPT_DIR, "integrated_debug.log")

SUBPROCESS_TIMEOUT: int = 180  # seconds

# Known Indian abbreviations (lowercase)
ABBREVIATIONS: Set[str] = {
    "blr", "del", "che", "kol", "pun", "hyd", "goa",
    "tn", "mh", "ka", "up", "ap", "ts", "mp",
    "rj", "gj", "pb", "hr", "wb", "jk", "uk", "hp",
}

# Preposition patterns that suggest a location follows
PREPOSITIONS = [
    " in ", " at ", " to ", " from ", " into ", " onto ",
    " here ", " there ", " inside ", " within ", " outside ",
    " over ", " under ", " above ", " below ", " beside ",
    " around ", " across ", " towards ", " through ", " along ",
    " past ", " off ", " down ", " up ", " out of ",
    " away from ", " nearby ", " far from ", " close to ",
    " in front of ", " on top of ", " in the middle of ",
    " in the heart of ", " in vicinity of ", " in and around ",
]

# Stderr keywords to suppress (from child model processes)
_SUPPRESSED_STDERR = (
    "legacy behaviour", "transformers",
    "max retries exceeded", "connection",
    "regex parse",
)

# ── Logging helper ─────────────────────────────────────────────────────────
logger = logging.getLogger("integrated")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s  %(message)s")


def log_debug(message: str) -> None:
    """Append a timestamped debug message to the log file."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.datetime.now()}: {message}\n")
    except Exception as exc:
        print(f"Log write error: {exc}", file=sys.stderr)


# ── Load reference data ───────────────────────────────────────────────────
log_debug("Loading reference data & spaCy model.")

try:
    nlp = spacy.load("en_core_web_sm")
    log_debug("spaCy model loaded.")
except Exception as exc:
    log_debug(f"CRITICAL: spaCy load failed – {exc}")
    print(f"ERROR: spaCy load failed. See {LOG_FILE}", file=sys.stderr)
    sys.exit(1)


def _load_city_list(csv_path: str) -> Tuple[Set[str], pd.DataFrame]:
    """Return (set_of_district_names_lower, full_dataframe)."""
    df = pd.read_csv(csv_path)
    cities = {str(c).strip().lower() for c in df["district"].dropna().unique()}
    return cities, df


CITY_CSV_PATH = os.path.join(SCRIPT_DIR, "final_training_geocoded.csv")
valid_cities, city_df = _load_city_list(CITY_CSV_PATH)


def _load_geonames() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load IN.txt and admin1CodesASCII.txt once; return (in_df, admin1_dict)."""
    in_txt_path = os.path.join(SCRIPT_DIR, "IN.txt")
    admin1_path = os.path.join(SCRIPT_DIR, "admin1CodesASCII.txt")

    in_df = pd.read_csv(
        in_txt_path, sep="\t", header=None, low_memory=False,
        names=[
            "geonameid", "name", "asciiname", "alternatenames",
            "latitude", "longitude", "feature_class", "feature_code",
            "country_code", "cc2", "admin1", "admin2", "admin3", "admin4",
            "population", "elevation", "dem", "timezone", "modification_date",
        ],
    )
    admin1 = pd.read_csv(
        admin1_path, sep="\t", header=None,
        names=["code", "name", "name_ascii", "geonameid"],
    )
    admin1_dict = {
        row["code"].split(".")[1]: row["name"]
        for _, row in admin1.iterrows()
        if row["code"].startswith("IN")
    }
    return in_df, admin1_dict


# Cache GeoNames data so it's loaded only once per process
_geonames_cache: Optional[Tuple[pd.DataFrame, Dict[str, str]]] = None


def _get_geonames() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Return cached (in_df, admin1_dict)."""
    global _geonames_cache
    if _geonames_cache is None:
        _geonames_cache = _load_geonames()
    return _geonames_cache


# ── Routing decision helpers ──────────────────────────────────────────────

def is_short_input(text: str) -> bool:
    """True when input has ≤ 2 alphabetic words."""
    words = text.strip().split()
    return len(words) <= 2 and all(w.isalpha() for w in words)


def has_abbreviation(text: str) -> bool:
    """True when any token is a known Indian abbreviation."""
    return any(tok in ABBREVIATIONS for tok in text.lower().split())


def has_phrase_after_preposition(text: str) -> bool:
    """True when text contains a spatial preposition."""
    lower = text.lower()
    return any(prep in lower for prep in PREPOSITIONS)


# ── Ambiguity detection ───────────────────────────────────────────────────

def check_city_ambiguity(city_name: str) -> bool:
    """Return True if *city_name* appears in more than one Indian state."""
    in_df, admin1_dict = _get_geonames()
    matched = in_df[in_df["asciiname"].str.lower() == city_name.lower()]
    states: Set[str] = set()
    for _, row in matched.iterrows():
        code = str(int(float(row["admin1"]))) if pd.notna(row["admin1"]) else "NA"
        state = admin1_dict.get(code)
        if state:
            states.add(state)
    return len(states) > 1


def print_ambiguous_cities(city_name: str) -> None:
    """Log all state occurrences for an ambiguous city name."""
    in_df, admin1_dict = _get_geonames()
    matched = in_df[in_df["asciiname"].str.lower() == city_name.lower()]

    print(f"\nInput is correct, routing to LLaMA3.", file=sys.stderr)
    print(f"\nAmbiguous city entries found for '{city_name.title()}':", file=sys.stderr)

    printed: Set[str] = set()
    for _, row in matched.iterrows():
        code = str(int(float(row["admin1"]))) if pd.notna(row["admin1"]) else "NA"
        state = admin1_dict.get(code, "NA")
        if state in printed:
            continue
        printed.add(state)
        print(
            f"  • {city_name.title()} – {state} "
            f"(lat: {row['latitude']}, lon: {row['longitude']})",
            file=sys.stderr,
        )

    print(
        f"\nAmbiguous city '{city_name.title()}' found in multiple states. "
        "Routing to LLaMA3.",
        file=sys.stderr,
    )


# ── Direct geocoding (exact, unambiguous match) ──────────────────────────

def direct_geocode(city_name: str) -> None:
    """Print JSON geocode result for an exact, unambiguous city match."""
    print(
        f"\nInput is correct. Directly geocoding '{city_name.title()}'.",
        file=sys.stderr,
    )
    row = city_df[city_df["district"].str.lower() == city_name.lower()].iloc[0]
    result = {
        "corrected_address": city_name.title(),
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "district": row["district"],
        "state": row["state"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


# ── Subprocess runner (single implementation) ─────────────────────────────

def _should_suppress_stderr(stderr_text: str) -> bool:
    """Return True if stderr only contains noise we want to hide."""
    lower = stderr_text.lower()
    return any(kw in lower for kw in _SUPPRESSED_STDERR)


def run_subprocess(script_path: str, text: str, route_reason: str) -> None:
    """
    Run a model script (*script_path*) with *text* as input and print
    the resulting JSON to stdout.

    All subprocess error-handling, timeout management, and stderr
    filtering is centralised here.
    """
    log_debug(f"Routing to {route_reason} – Script: {script_path}")
    print(f"Routing to {route_reason} (Input: '{text}')", file=sys.stderr)

    try:
        proc = subprocess.run(
            [sys.executable, script_path, text],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=SUBPROCESS_TIMEOUT,
        )

        # Filtered stderr logging
        if proc.stderr and not _should_suppress_stderr(proc.stderr):
            log_debug(f"stderr [{os.path.basename(script_path)}]:\n{proc.stderr.strip()}")
            print(
                f"stderr [{os.path.basename(script_path)}]:\n{proc.stderr.strip()}",
                file=sys.stderr,
            )

        if proc.returncode != 0:
            log_debug(
                f"'{os.path.basename(script_path)}' exited with code {proc.returncode}."
            )

        # Extract last JSON object from stdout
        json_matches = re.findall(
            r"\{(?:[^{}]*|\{[^{}]*\})*\}", proc.stdout, re.DOTALL
        )
        if json_matches:
            log_debug(f"JSON extracted from {os.path.basename(script_path)}.")
            print(json_matches[-1])
        else:
            log_debug(
                f"No JSON from {os.path.basename(script_path)}. "
                f"Raw stdout: {proc.stdout.strip()}"
            )
            print(json.dumps(_na_result(script_path, "No JSON")))

    except FileNotFoundError:
        log_debug(f"Script not found: {script_path}")
        print(f"Error: Script not found: {script_path}", file=sys.stderr)
        print(json.dumps(_na_result(script_path, "Script Not Found")))

    except subprocess.TimeoutExpired:
        log_debug(f"Timeout for '{os.path.basename(script_path)}' on input: '{text}'")
        print(f"Error: '{os.path.basename(script_path)}' timed out.", file=sys.stderr)
        print(json.dumps(_na_result(script_path, "Timeout")))

    except Exception as exc:
        log_debug(f"Unexpected error routing to {route_reason}: {exc}")
        print(f"Error: {route_reason}: {exc}", file=sys.stderr)
        print(json.dumps(_na_result(script_path, "Unexpected Error")))


def _na_result(script_path: str, reason: str) -> Dict[str, str]:
    """Build a fallback NA result dict."""
    return {
        "corrected_address": "NA",
        "latitude": "NA",
        "longitude": "NA",
        "model_used": f"{os.path.basename(script_path)} – {reason}",
    }


# ── Main routing logic ───────────────────────────────────────────────────

def route_text(text: str) -> None:
    """
    Determine the best processing strategy for *text* and execute it.

    Priority order:
        1. Exact city match  →  direct geocode (or LLaMA3 if ambiguous)
        2. Short / abbreviation / preposition  →  T5
        3. Everything else  →  LLaMA3
    """
    log_debug(f"Input: {text}")

    # ── Step 1: check for exact one-to-one city match ──
    words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
    for word in words:
        if word in valid_cities:
            log_debug(f"Exact city match: {word}")
            if check_city_ambiguity(word):
                print_ambiguous_cities(word)
                run_subprocess(LLAMA3_SCRIPT_PATH, text, "LLaMA3 (ambiguity)")
            else:
                direct_geocode(word)
            return

    # ── Step 2: rule-based routing ──
    if is_short_input(text) or has_abbreviation(text) or has_phrase_after_preposition(text):
        run_subprocess(T5_SCRIPT_PATH, text, "T5 (short / abbreviation / preposition)")
    else:
        run_subprocess(LLAMA3_SCRIPT_PATH, text, "LLaMA3 (contextual inference)")


# ── Entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    log_debug("integrated.py started.")

    if not sys.stdin.isatty():
        user_input = sys.stdin.read().strip()
    elif len(sys.argv) > 1:
        user_input = sys.argv[1].strip()
    else:
        user_input = input("Enter your text: ").strip()

    if user_input:
        route_text(user_input)
    else:
        log_debug("No input provided.")
        print("{}", file=sys.stderr)

    log_debug("integrated.py finished.")
    sys.exit(0)
