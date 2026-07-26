"""
extract_gas.py

Reads a single-page statement PDF (table-format transaction records),
transcribes every row using a local Qwen2.5-VL model via Ollama, then
classifies each row as gas/fuel-related using a second, lightweight
local text model's judgment (no hardcoded keyword list).

Usage:
    python3 extract_gas.py february_statement.pdf

Requirements:
    pip install pdf2image requests tqdm --break-system-packages
    brew install poppler
    ollama pull qwen2.5vl:7b   # vision model, reads the statement image
    ollama pull qwen2.5:7b     # small text model, classifies each row
"""

import sys
print("[DEBUG L18] imported sys")
import base64
print("[DEBUG L20] imported base64")
import json
print("[DEBUG L22] imported json")
import time
print("[DEBUG L24] imported time")
from io import BytesIO
print("[DEBUG L26] imported BytesIO")

import requests
print("[DEBUG L28] imported requests")
from pdf2image import convert_from_path
print("[DEBUG L30] imported convert_from_path")
from tqdm import tqdm
print("[DEBUG L32] imported tqdm")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VISION_MODEL = "qwen2.5vl:7b"
print("[DEBUG L39] set VISION_MODEL")

# A small text-only model is enough for a one-line classification judgment
# and runs much faster than the vision model since there's no image to
# process. Swap for whatever lightweight text model you have pulled.
CLASSIFY_MODEL = "qwen2.5:7b"
print("[DEBUG L44] set CLASSIFY_MODEL")

OLLAMA_URL = "http://localhost:11434/api/generate"
print("[DEBUG L47] set OLLAMA_URL")

PDF_DPI = 150
print("[DEBUG L50] set PDF_DPI")
MAX_IMAGE_WIDTH = 1000
print("[DEBUG L52] set MAX_IMAGE_WIDTH")

KEEP_ALIVE = "30m"
print("[DEBUG L55] set KEEP_ALIVE")
REQUEST_TIMEOUT = 600
print("[DEBUG L57] set REQUEST_TIMEOUT")

# Row is treated as a gas transaction if the model's confidence is at or
# above this threshold. Lower it to catch more borderline rows (at the
# cost of more false positives), raise it to be stricter.
GAS_CONFIDENCE_THRESHOLD = 0.5
print("[DEBUG L62] set GAS_CONFIDENCE_THRESHOLD")

# Row schema keys -- used to detect when the transcription model returns a
# single row object directly instead of wrapping it (or an array of them)
# in a list.
ROW_KEYS = {"date", "merchant", "amount"}
print("[DEBUG L68] set ROW_KEYS")

TRANSCRIBE_PROMPT = """This image shows a page from a bank/credit card statement, \
formatted as a table of transaction records.

Transcribe EVERY row in the table exactly as shown. Do not skip any row, \
even ones you are unsure about or that look unusual. Do not filter or judge \
which rows matter -- include all of them, not just gas-related ones.

You MUST return a JSON ARRAY containing one object per row, even if there is \
only one row, and even if there are many rows (10, 20, 50+). Never return a \
single bare object -- always wrap every row inside an array.

Return ONLY JSON, no other text, in this exact format:
{"rows": [{"date": "", "merchant": "", "amount": ""}, {"date": "", "merchant": "", "amount": ""}]}

If the page has no table rows, return: {"rows": []}"""
print("[DEBUG L84] set TRANSCRIBE_PROMPT")

CLASSIFY_PROMPT_TEMPLATE = """You are reviewing one line item from a credit card statement.

Date: {date}
Merchant description: {merchant}
Amount: {amount}

Question: how likely is it that this transaction is a purchase of gasoline \
or fuel at a gas station? Consider the merchant name -- known gas station \
brands (Shell, Esso, Petro-Canada, Chevron, Exxon, Mobil, Circle K, Husky, \
Irving, Ultramar, Sunoco, etc.), generic terms like "gas" or "fuel", and \
whether the description looks like something else entirely (restaurant, \
retail, subscription, e-transfer, etc.).

Respond with ONLY a JSON object in this exact format:
{{"confidence": <integer 0-100>}}

Where 0 means definitely not a gas/fuel purchase and 100 means certainly is."""
print("[DEBUG L101] set CLASSIFY_PROMPT_TEMPLATE")


# ---------------------------------------------------------------------------
# Helpers -- PDF / vision transcription
# ---------------------------------------------------------------------------

def encode_image(img) -> str:
    print("[DEBUG L109] encode_image: enter")
    if img.width > MAX_IMAGE_WIDTH:
        print("[DEBUG L111] encode_image: resizing")
        ratio = MAX_IMAGE_WIDTH / img.width
        print("[DEBUG L113] encode_image: computed ratio")
        img = img.resize((MAX_IMAGE_WIDTH, int(img.height * ratio)))
        print("[DEBUG L115] encode_image: resized")
    buf = BytesIO()
    print("[DEBUG L117] encode_image: created BytesIO")
    img.save(buf, format="PNG")
    print("[DEBUG L119] encode_image: saved PNG to buffer")
    result = base64.b64encode(buf.getvalue()).decode("utf-8")
    print("[DEBUG L121] encode_image: encoded base64")
    return result


def _extract_rows_from_parsed(parsed):
    """
    Normalize whatever JSON shape the model returned into a list of row
    dicts. Handles:
      - a bare list:                [{"date": ...}, ...]
      - an object wrapping a list:  {"rows": [...]} / {"gas_transactions": [...]}
      - a single row returned bare: {"date": ..., "merchant": ..., "amount": ...}
    """
    print("[DEBUG L131] _extract_rows_from_parsed: enter")

    if isinstance(parsed, list):
        print("[DEBUG L134] _extract_rows_from_parsed: parsed is already a list")
        return parsed

    if isinstance(parsed, dict):
        print("[DEBUG L138] _extract_rows_from_parsed: parsed is a dict")

        if ROW_KEYS.issubset(set(parsed.keys())):
            print("[DEBUG L141] _extract_rows_from_parsed: dict looks like a single row, wrapping in list")
            return [parsed]

        for key, value in parsed.items():
            if isinstance(value, list):
                print(f"[DEBUG L146] _extract_rows_from_parsed: found list under key '{key}'")
                return value

        print("[DEBUG L149] _extract_rows_from_parsed: dict had no list and isn't a single row, giving up")
        return []

    print("[DEBUG L152] _extract_rows_from_parsed: parsed is neither list nor dict")
    return []


def transcribe_page(img_b64: str) -> list:
    print("[DEBUG L157] transcribe_page: enter")
    payload = {
        "model": VISION_MODEL,
        "prompt": TRANSCRIBE_PROMPT,
        "images": [img_b64],
        "stream": False,
        "format": "json",
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.0},
    }
    print("[DEBUG L166] transcribe_page: built payload")
    print("[DEBUG L167] transcribe_page: about to POST to Ollama (may hang here)...")
    resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    print("[DEBUG L169] transcribe_page: got response")
    resp.raise_for_status()
    print("[DEBUG L171] transcribe_page: status OK")
    raw = resp.json()["response"]
    print("[DEBUG L173] transcribe_page: parsed response envelope, raw model text extracted")

    try:
        print("[DEBUG L176] transcribe_page: entering json.loads")
        parsed = json.loads(raw)
        print("[DEBUG L178] transcribe_page: json.loads succeeded")
    except json.JSONDecodeError:
        print(f"[warn] could not parse model output, raw response:\n{raw[:500]}")
        print("[DEBUG L181] transcribe_page: JSONDecodeError, returning []")
        return []

    rows = _extract_rows_from_parsed(parsed)
    print(f"[DEBUG L185] transcribe_page: normalized to {len(rows)} row(s)")

    if not rows:
        print(f"[warn] no rows extracted, raw response:\n{raw[:500]}")

    return rows


# ---------------------------------------------------------------------------
# Helpers -- model-based row classification (replaces keyword matching)
# ---------------------------------------------------------------------------

def classify_row(row: dict) -> float:
    """
    Ask a local text model to judge how likely this row is a gas/fuel
    purchase, based on its actual understanding of merchant names --
    no hardcoded keyword list. Returns a confidence float in [0.0, 1.0].
    Falls back to 0.0 on any parsing/request failure so one bad row
    doesn't crash the whole run.
    """
    print("[DEBUG L199] classify_row: enter, row =", row)
    prompt = CLASSIFY_PROMPT_TEMPLATE.format(
        date=row.get("date", ""),
        merchant=row.get("merchant", ""),
        amount=row.get("amount", ""),
    )
    print("[DEBUG L205] classify_row: built prompt")

    payload = {
        "model": CLASSIFY_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.0},
    }
    print("[DEBUG L214] classify_row: built payload")

    try:
        print("[DEBUG L217] classify_row: about to POST to Ollama")
        resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        print("[DEBUG L219] classify_row: got response")
        resp.raise_for_status()
        print("[DEBUG L221] classify_row: status OK")
    except requests.exceptions.RequestException as e:
        print(f"[warn] classify_row: request failed: {e}")
        print("[DEBUG L224] classify_row: returning 0.0 due to request error")
        return 0.0

    raw = resp.json()["response"]
    print("[DEBUG L228] classify_row: extracted raw model text")

    try:
        parsed = json.loads(raw)
        print("[DEBUG L231] classify_row: json.loads succeeded, parsed =", parsed)
        confidence_raw = float(parsed.get("confidence", 0))
        print("[DEBUG L233] classify_row: extracted confidence_raw =", confidence_raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        print(f"[warn] classify_row: could not parse confidence, raw response:\n{raw[:300]}")
        print("[DEBUG L236] classify_row: returning 0.0 due to parse error")
        return 0.0

    confidence = max(0.0, min(confidence_raw / 100.0, 1.0))
    print("[DEBUG L239] classify_row: normalized confidence =", confidence)
    return confidence


def parse_amount(value) -> float:
    """
    Convert an amount field (which may be a string like '$45.20', '45.20',
    or already a number) into a float. Returns 0.0 if it can't be parsed,
    so a bad row doesn't crash the summary but is still visible in the
    per-row breakdown for manual review.
    """
    print("[DEBUG L248] parse_amount: enter, value =", value)
    if value is None:
        print("[DEBUG L250] parse_amount: value is None, returning 0.0")
        return 0.0
    if isinstance(value, (int, float)):
        print("[DEBUG L253] parse_amount: value already numeric")
        return float(value)
    cleaned = str(value).replace("$", "").replace(",", "").strip()
    print("[DEBUG L256] parse_amount: cleaned string =", cleaned)
    try:
        result = float(cleaned)
        print("[DEBUG L259] parse_amount: parsed successfully:", result)
        return result
    except ValueError:
        print("[DEBUG L262] parse_amount: could not parse, returning 0.0")
        return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[DEBUG L270] main: enter")
    if len(sys.argv) < 2:
        print("Usage: python3 extract_gas.py <statement.pdf>")
        print("[DEBUG L273] main: missing argv, exiting")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print("[DEBUG L277] main: got pdf_path:", pdf_path)

    print(f"Converting {pdf_path} to an image (dpi={PDF_DPI})...")
    print("[DEBUG L280] main: about to convert_from_path")
    pages = convert_from_path(pdf_path, dpi=PDF_DPI)
    print("[DEBUG L282] main: convert_from_path done")
    if not pages:
        print("No pages found in PDF.")
        print("[DEBUG L285] main: no pages, exiting")
        sys.exit(1)
    page = pages[0]
    print("[DEBUG L288] main: got first page:", page)

    print("Encoding image and sending to vision model for transcription...")
    print("[DEBUG L291] main: about to encode_image")
    img_b64 = encode_image(page)
    print("[DEBUG L293] main: encode_image done")

    start = time.time()
    print("[DEBUG L296] main: set start time")
    try:
        print("[DEBUG L298] main: about to transcribe_page")
        rows = transcribe_page(img_b64)
        print("[DEBUG L300] main: transcribe_page done")
    except requests.exceptions.Timeout:
        print(f"[error] transcription request timed out after {REQUEST_TIMEOUT}s")
        print("[DEBUG L303] main: timeout, exiting")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"[error] transcription request failed: {e}")
        print("[DEBUG L307] main: request error, exiting")
        sys.exit(1)
    transcribe_elapsed = time.time() - start
    print("[DEBUG L310] main: computed transcribe_elapsed")

    print(f"\n{len(rows)} row(s) transcribed in {transcribe_elapsed:.1f}s")
    print("Classifying each row with the local text model...")

    classify_start = time.time()
    for row in tqdm(rows, desc="Classifying rows", unit="row"):
        row["confidence"] = classify_row(row)
    classify_elapsed = time.time() - classify_start
    print(f"[DEBUG L319] main: classified {len(rows)} row(s) in {classify_elapsed:.1f}s")

    gas_rows = [r for r in rows if r["confidence"] >= GAS_CONFIDENCE_THRESHOLD]
    gas_rows.sort(key=lambda r: r["confidence"], reverse=True)
    print("[DEBUG L323] main: filtered and sorted gas_rows")

    total_elapsed = transcribe_elapsed + classify_elapsed
    print(f"\n{len(gas_rows)} gas row(s) found (threshold={GAS_CONFIDENCE_THRESHOLD}) -- total {total_elapsed:.1f}s")
    print("[DEBUG L327] main: printed summary")

    print("[DEBUG L329] main: opening february_all_rows.json")
    with open("february_all_rows.json", "w") as f:
        print("[DEBUG L331] main: writing all rows")
        json.dump(rows, f, indent=2)
        print("[DEBUG L333] main: wrote all rows")

    print("[DEBUG L335] main: opening february_gas_receipts.json")
    with open("february_gas_receipts.json", "w") as f:
        print("[DEBUG L337] main: writing gas rows")
        json.dump(gas_rows, f, indent=2)
        print("[DEBUG L339] main: wrote gas rows")

    print("[DEBUG L341] main: opening summary.txt")
    gas_total = sum(parse_amount(r.get("amount")) for r in gas_rows)
    print("[DEBUG L343] main: computed gas_total:", gas_total)
    with open("summary.txt", "w") as f:
        print("[DEBUG L345] main: writing summary header")
        f.write(f"Source PDF: {pdf_path}\n")
        f.write(f"Total rows transcribed: {len(rows)}\n")
        f.write(f"Gas transactions found (confidence >= {GAS_CONFIDENCE_THRESHOLD * 100:.0f}%): {len(gas_rows)}\n")
        f.write(f"Transcription time: {transcribe_elapsed:.1f}s\n")
        f.write(f"Classification time: {classify_elapsed:.1f}s\n")
        print("[DEBUG L351] main: wrote header lines")

        f.write("\nGas transactions:\n")
        print("[DEBUG L354] main: writing gas transaction breakdown")
        for r in gas_rows:
            date = r.get("date", "")
            merchant = r.get("merchant", "")
            amount = parse_amount(r.get("amount"))
            confidence = r.get("confidence", 0.0)
            f.write(f"  {date} | {merchant} | ${amount:.2f} | confidence: {confidence * 100:.0f}%\n")
        print("[DEBUG L361] main: wrote all gas transaction lines")

        f.write(f"\nTotal gas cost: ${gas_total:.2f}\n")
        print("[DEBUG L364] main: wrote total gas cost line")

    print("Output files: february_all_rows.json, february_gas_receipts.json, summary.txt")
    print("[DEBUG L368] main: done")


if __name__ == "__main__":
    print("[DEBUG L372] calling main()")
    main()
    print("[DEBUG L374] main() returned")