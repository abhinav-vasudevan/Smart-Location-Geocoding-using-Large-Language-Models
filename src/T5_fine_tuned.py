"""
T5_fine_tuned.py — T5-based address correction and geocoding engine.

This script:
1. Extracts a location fragment from the input sentence (via POS tagging).
2. Runs it through a fine-tuned T5 model to correct spelling / abbreviations.
3. Enriches the corrected text with lat/lon, district, and state from GeoNames.
4. Detects ambiguous cities and falls back to LLaMA3 when needed.
5. Persists every correction to PostgreSQL.

Usage:
    python T5_fine_tuned.py "near blr"
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
import subprocess
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import nltk
import pandas as pd
import psycopg2
import torch
from nltk.tag import pos_tag
from nltk.tokenize import TreebankWordTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR: str = os.path.join(SCRIPT_DIR, "t5_corrector_final")
NLTK_DATA_DIR: str = os.path.join(SCRIPT_DIR, "nltk_data")
IN_TXT_PATH: str = os.path.join(SCRIPT_DIR, "IN.txt")
ADMIN1_PATH: str = os.path.join(SCRIPT_DIR, "admin1CodesASCII.txt")
ADMIN2_PATH: str = os.path.join(SCRIPT_DIR, "admin2Codes.txt")
LLAMA3_SCRIPT: str = os.path.join(SCRIPT_DIR, "Context_mapping_LLAMA3.py")

# ── Database configuration ─────────────────────────────────────────────────
DB_CONFIG: Dict[str, str] = {
    "dbname": os.getenv("DB_NAME", "Address_corrector"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

# ── NLTK data ──────────────────────────────────────────────────────────────
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

# ── Custom abbreviation map ────────────────────────────────────────────────
CUSTOM_ABBREVIATIONS: Dict[str, str] = {
    "BLR": "Bangalore", "DEL": "Delhi", "BOM": "Mumbai",
    "HYD": "Hyderabad", "CHE": "Chennai", "KOL": "Kolkata",
    "PUN": "Pune", "GOA": "Goa", "TN": "Tamil Nadu",
    "MH": "Maharashtra", "KA": "Karnataka", "UP": "Uttar Pradesh",
    "AP": "Andhra Pradesh", "TS": "Telangana", "MP": "Madhya Pradesh",
    "RJ": "Rajasthan", "GJ": "Gujarat", "PB": "Punjab",
    "HR": "Haryana", "WB": "West Bengal", "JK": "Jammu and Kashmir",
    "UK": "Uttarakhand", "HP": "Himachal Pradesh",
}

# Spatial prepositions used to extract location fragments
_LOCATION_PREPS = {
    "in", "at", "near", "from", "into", "onto", "here", "there",
    "inside", "within", "outside", "over", "under", "above",
    "below", "beside", "around", "across", "towards", "through", "along",
}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_geonames() -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """Load GeoNames files and return (in_df, admin1_dict, admin2_dict)."""
    in_df = pd.read_csv(
        IN_TXT_PATH, sep="\t", header=None, low_memory=False,
        names=[
            "geonameid", "name", "asciiname", "alternatenames",
            "latitude", "longitude", "feature_class", "feature_code",
            "country_code", "cc2", "admin1", "admin2", "admin3", "admin4",
            "population", "elevation", "dem", "timezone", "modification_date",
        ],
    )

    admin1 = pd.read_csv(
        ADMIN1_PATH, sep="\t", header=None,
        names=["code", "name", "name_ascii", "geonameid"],
    )
    admin1_dict: Dict[str, str] = {
        row["code"].split(".")[1]: row["name"]
        for _, row in admin1.iterrows()
        if row["code"].startswith("IN")
    }

    admin2 = pd.read_csv(
        ADMIN2_PATH, sep="\t", header=None,
        names=["code", "name", "name_ascii", "geonameid"],
    )
    admin2_dict: Dict[str, str] = {
        f"{parts[1]}.{parts[2]}": row["name"]
        for _, row in admin2.iterrows()
        if (parts := row["code"].split(".")) and len(parts) == 3 and parts[0] == "IN"
    }

    return in_df, admin1_dict, admin2_dict


def _load_t5_model() -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """Load the fine-tuned T5 model and tokenizer from disk."""
    if not os.path.exists(MODEL_DIR):
        print(f"Error: model directory '{MODEL_DIR}' not found.", file=sys.stderr)
        sys.exit(1)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    tokenizer = T5Tokenizer(
        vocab_file=os.path.join(MODEL_DIR, "spiece.model"), extra_ids=100
    )
    return model, tokenizer


# ── Eagerly load everything once at import time ──
in_df, admin1_dict, admin2_dict = _load_geonames()
model, tokenizer = _load_t5_model()


# ═══════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════

def normalize_tokens(tokens: List[str]) -> List[str]:
    """Expand known abbreviations inside a token list."""
    return [CUSTOM_ABBREVIATIONS.get(t.upper().strip(), t) for t in tokens]


def predict(noisy_input: str) -> str:
    """Run the T5 model on *noisy_input* and return the corrected string."""
    input_ids = tokenizer(noisy_input, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=128)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def fuzzy_lookup(code_dict: Dict[str, str], target_name: str) -> str:
    """Find the best fuzzy match in *code_dict* values for *target_name*."""
    best_score = 0.0
    best_match = "NA"
    for _, name in code_dict.items():
        score = SequenceMatcher(None, target_name.lower(), name.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = name
    return best_match if best_score > 0.5 else "NA"


def enrich_address(
    corrected_address: str,
) -> Tuple[Optional[str], Any, Any, str, str]:
    """
    Match *corrected_address* tokens against GeoNames and return
    (ascii_name, latitude, longitude, district, state).
    """
    tokens = normalize_tokens([t.strip() for t in corrected_address.split(",")])
    best_score = 0.0
    best_row = None

    for _, row in in_df.iterrows():
        name = str(row.get("asciiname", "")).strip()
        if not name or name.lower() == "nan":
            continue
        for token in tokens:
            score = SequenceMatcher(None, token.lower(), name.lower()).ratio()
            if score > best_score:
                best_score = score
                best_row = row

    if best_row is not None and best_score > 0.6:
        lat = best_row["latitude"]
        lon = best_row["longitude"]

        try:
            a1_code = str(int(float(best_row.get("admin1", "")))) if pd.notna(best_row.get("admin1")) else "NA"
        except (ValueError, TypeError):
            a1_code = "NA"
        try:
            a2_code = str(int(float(best_row.get("admin2", "")))) if pd.notna(best_row.get("admin2")) else "NA"
        except (ValueError, TypeError):
            a2_code = "NA"

        state = admin1_dict.get(a1_code, fuzzy_lookup(admin1_dict, best_row["asciiname"]))
        district = admin2_dict.get(
            f"{a1_code}.{a2_code}", fuzzy_lookup(admin2_dict, best_row["asciiname"])
        )
        return best_row["asciiname"], lat, lon, district, state

    return None, "NA", "NA", "NA", "NA"


def extract_location_from_sentence(sentence: str) -> str:
    """
    Use POS-tagging to pull the location fragment that appears after a
    spatial preposition (e.g. "I'm going to *Bangalore*").
    Falls back to the whole sentence if no preposition is found.
    """
    tok = TreebankWordTokenizer()
    tokens = tok.tokenize(sentence)
    tagged = pos_tag(tokens)

    for i, (word, _) in enumerate(tagged):
        if word.lower() in _LOCATION_PREPS:
            location_words: List[str] = []
            j = i + 1
            while j < len(tagged) and tagged[j][1] in ("NN", "NNP", "NNS", ",", "CC"):
                location_words.append(tagged[j][0])
                j += 1
            if location_words:
                return " ".join(location_words).replace(",", "").strip()

    return sentence.strip()


def check_ambiguity(city_name: str) -> List[Tuple[str, float, float]]:
    """Return list of (state, lat, lon) for all states where *city_name* exists."""
    if not isinstance(city_name, str):
        return []
    matched = in_df[in_df["asciiname"].str.lower() == city_name.lower()]
    seen_states: set = set()
    results: List[Tuple[str, float, float]] = []
    for _, row in matched.iterrows():
        a1 = str(int(float(row["admin1"]))) if pd.notna(row["admin1"]) else "NA"
        state = admin1_dict.get(a1)
        if state and state not in seen_states:
            seen_states.add(state)
            results.append((state, row["latitude"], row["longitude"]))
    return results


def _is_valid(lat: Any, lon: Any) -> bool:
    """Return True if lat/lon are usable numeric values."""
    return lat not in ("NA", None) and lon not in ("NA", None)


# ── Database persistence ──────────────────────────────────────────────────

def save_to_db(input_text: str, output_text: str, lat: Any, lon: Any) -> None:
    """Insert a correction record into PostgreSQL."""
    try:
        lat = float(lat) if pd.notna(lat) and str(lat).replace(".", "", 1).isdigit() else None
        lon = float(lon) if pd.notna(lon) and str(lon).replace(".", "", 1).isdigit() else None

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO corrections (input_text, model_used, output_text, latitude, longitude)
               VALUES (%s, %s, %s, %s, %s);""",
            (input_text, "T5", output_text, lat, lon),
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
    """Entry-point: read input, correct, geocode, handle ambiguity, persist."""
    # ── Read input ──
    if not sys.stdin.isatty():
        sentence = sys.stdin.read().strip()
    elif len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:]).strip()
    else:
        sentence = input("Enter a sentence: ").strip()

    # ── Step 1: extract location fragment & run T5 ──
    noisy_address = extract_location_from_sentence(sentence)
    model_corrected = predict(noisy_address)

    # ── Step 2: enrich both raw & corrected address ──
    ascii_raw, lat_raw, lon_raw, dist_raw, state_raw = enrich_address(noisy_address)
    ascii_t5, lat_t5, lon_t5, dist_t5, state_t5 = enrich_address(model_corrected)

    # Prefer raw match; fall back to T5-corrected match
    if _is_valid(lat_raw, lon_raw):
        final_corrected, lat, lon, district, state = ascii_raw, lat_raw, lon_raw, dist_raw, state_raw
    elif _is_valid(lat_t5, lon_t5):
        final_corrected, lat, lon, district, state = ascii_t5, lat_t5, lon_t5, dist_t5, state_t5
    else:
        final_corrected = model_corrected
        lat = lon = district = state = "NA"

    # ── Step 3: output JSON result ──
    print(json.dumps({
        "corrected_address": final_corrected,
        "latitude": lat,
        "longitude": lon,
        "district": district,
        "state": state,
    }, ensure_ascii=False, indent=2))

    # ── Step 4: handle ambiguity → optional LLaMA3 fallback ──
    ambiguities = check_ambiguity(final_corrected)
    if len(ambiguities) > 1:
        print(f"\nAmbiguous city entries for '{final_corrected}':", file=sys.stderr)
        for state_name, la, lo in ambiguities:
            print(f"  • {final_corrected} – {state_name} (lat: {la}, lon: {lo})", file=sys.stderr)
        print(f"\nRouting to LLaMA3 for disambiguation.", file=sys.stderr)
        try:
            proc = subprocess.run(
                [sys.executable, LLAMA3_SCRIPT, sentence],
                capture_output=True, text=True, check=True,
                encoding="utf-8", errors="replace",
            )
            print("\nLLaMA3 Output:")
            print(proc.stdout.strip())
        except subprocess.CalledProcessError as exc:
            print(f"LLaMA3 failed (code {exc.returncode}):", file=sys.stderr)
            print("stdout:", exc.stdout.strip(), file=sys.stderr)
            print("stderr:", exc.stderr.strip(), file=sys.stderr)
        except FileNotFoundError:
            print(f"LLaMA3 script not found at {LLAMA3_SCRIPT}.", file=sys.stderr)
        except Exception as exc:
            print(f"LLaMA3 call failed: {exc}", file=sys.stderr)

    # ── Step 5: persist to database ──
    save_to_db(sentence, final_corrected, lat, lon)


if __name__ == "__main__":
    main()
