"""
extract_gas.py

Reads a single-page statement PDF (table-format transaction records),
transcribes every row using a local Qwen2.5-VL model via Ollama, then
filters for gas/fuel-related rows in Python (deterministic filtering,
not left up to the model).

Usage:
    python3 extract_gas.py february_statement.pdf

Requirements:
    pip install pdf2image requests --break-system-packages
    brew install poppler
    ollama pull qwen2.5vl:7b
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "qwen2.5vl:7b"
print("[DEBUG L37] set MODEL")
OLLAMA_URL = "http://localhost:11434/api/generate"
print("[DEBUG L39] set OLLAMA_URL")

PDF_DPI = 150
print("[DEBUG L42] set PDF_DPI")
MAX_IMAGE_WIDTH = 1000
print("[DEBUG L44] set MAX_IMAGE_WIDTH")

KEEP_ALIVE = "30m"
print("[DEBUG L47] set KEEP_ALIVE")
REQUEST_TIMEOUT = 600
print("[DEBUG L49] set REQUEST_TIMEOUT")

GAS_KEYWORDS = [
    "SHELL", "ESSO", "PETRO-CANADA", "PETRO CANADA", "PETROCAN",
    "CHEVRON", "EXXON", "MOBIL", "CIRCLE K", "HUSKY", "IRVING",
    "ULTRAMAR", "GAS", "FUEL", "SUNOCO", "7-ELEVEN FUEL",
]
print("[DEBUG L56] set GAS_KEYWORDS")

# Row schema keys -- used to detect when the model returns a single row
# object directly instead of wrapping it (or an array of them) in a list.
ROW_KEYS = {"date", "merchant", "amount"}
print("[DEBUG L61] set ROW_KEYS")

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
print("[DEBUG L78] set TRANSCRIBE_PROMPT")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_image(img) -> str:
    print("[DEBUG L86] encode_image: enter")
    if img.width > MAX_IMAGE_WIDTH:
        print("[DEBUG L88] encode_image: resizing")
        ratio = MAX_IMAGE_WIDTH / img.width
        print("[DEBUG L90] encode_image: computed ratio")
        img = img.resize((MAX_IMAGE_WIDTH, int(img.height * ratio)))
        print("[DEBUG L92] encode_image: resized")
    buf = BytesIO()
    print("[DEBUG L94] encode_image: created BytesIO")
    img.save(buf, format="PNG")
    print("[DEBUG L96] encode_image: saved PNG to buffer")
    result = base64.b64encode(buf.getvalue()).decode("utf-8")
    print("[DEBUG L98] encode_image: encoded base64")
    return result


def _extract_rows_from_parsed(parsed):
    """
    Normalize whatever JSON shape the model returned into a list of row
    dicts. Handles:
      - a bare list:                [{"date": ...}, ...]
      - an object wrapping a list:  {"rows": [...]} / {"gas_transactions": [...]}
      - a single row returned bare: {"date": ..., "merchant": ..., "amount": ...}
    """
    print("[DEBUG L108] _extract_rows_from_parsed: enter")

    if isinstance(parsed, list):
        print("[DEBUG L111] _extract_rows_from_parsed: parsed is already a list")
        return parsed

    if isinstance(parsed, dict):
        print("[DEBUG L115] _extract_rows_from_parsed: parsed is a dict")

        # Case: model returned exactly one row, unwrapped
        if ROW_KEYS.issubset(set(parsed.keys())):
            print("[DEBUG L119] _extract_rows_from_parsed: dict looks like a single row, wrapping in list")
            return [parsed]

        # Case: model wrapped the array under some key (rows, transactions, etc.)
        for key, value in parsed.items():
            if isinstance(value, list):
                print(f"[DEBUG L125] _extract_rows_from_parsed: found list under key '{key}'")
                return value

        print("[DEBUG L128] _extract_rows_from_parsed: dict had no list and isn't a single row, giving up")
        return []

    print("[DEBUG L131] _extract_rows_from_parsed: parsed is neither list nor dict")
    return []


def transcribe_page(img_b64: str) -> list:
    print("[DEBUG L136] transcribe_page: enter")
    payload = {
        "model": MODEL,
        "prompt": TRANSCRIBE_PROMPT,
        "images": [img_b64],
        "stream": False,
        "format": "json",
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.0},
    }
    print("[DEBUG L146] transcribe_page: built payload")
    print("[DEBUG L147] transcribe_page: about to POST to Ollama (may hang here)...")
    resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    print("[DEBUG L149] transcribe_page: got response")
    resp.raise_for_status()
    print("[DEBUG L151] transcribe_page: status OK")
    raw = resp.json()["response"]
    print("[DEBUG L153] transcribe_page: parsed response envelope, raw model text extracted")

    try:
        print("[DEBUG L156] transcribe_page: entering json.loads")
        parsed = json.loads(raw)
        print("[DEBUG L158] transcribe_page: json.loads succeeded")
    except json.JSONDecodeError:
        print(f"[warn] could not parse model output, raw response:\n{raw[:500]}")
        print("[DEBUG L161] transcribe_page: JSONDecodeError, returning []")
        return []

    rows = _extract_rows_from_parsed(parsed)
    print(f"[DEBUG L165] transcribe_page: normalized to {len(rows)} row(s)")

    if not rows:
        print(f"[warn] no rows extracted, raw response:\n{raw[:500]}")

    return rows


def is_gas_row(row: dict) -> bool:
    print("[DEBUG L174] is_gas_row: enter")
    text = " ".join(str(v) for v in row.values() if v).upper()
    print("[DEBUG L176] is_gas_row: built text")
    result = any(kw in text for kw in GAS_KEYWORDS)
    print("[DEBUG L178] is_gas_row: computed result")
    return result


def parse_amount(value) -> float:
    """
    Convert an amount field (which may be a string like '$45.20', '45.20',
    or already a number) into a float. Returns 0.0 if it can't be parsed,
    so a bad row doesn't crash the summary but is still visible in the
    per-row breakdown for manual review.
    """
    print("[DEBUG L182] parse_amount: enter, value =", value)
    if value is None:
        print("[DEBUG L184] parse_amount: value is None, returning 0.0")
        return 0.0
    if isinstance(value, (int, float)):
        print("[DEBUG L187] parse_amount: value already numeric")
        return float(value)
    cleaned = str(value).replace("$", "").replace(",", "").strip()
    print("[DEBUG L190] parse_amount: cleaned string =", cleaned)
    try:
        result = float(cleaned)
        print("[DEBUG L193] parse_amount: parsed successfully:", result)
        return result
    except ValueError:
        print("[DEBUG L196] parse_amount: could not parse, returning 0.0")
        return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[DEBUG L186] main: enter")
    if len(sys.argv) < 2:
        print("Usage: python3 extract_gas.py <statement.pdf>")
        print("[DEBUG L190] main: missing argv, exiting")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print("[DEBUG L194] main: got pdf_path:", pdf_path)

    print(f"Converting {pdf_path} to an image (dpi={PDF_DPI})...")
    print("[DEBUG L197] main: about to convert_from_path")
    pages = convert_from_path(pdf_path, dpi=PDF_DPI)
    print("[DEBUG L199] main: convert_from_path done")
    if not pages:
        print("No pages found in PDF.")
        print("[DEBUG L202] main: no pages, exiting")
        sys.exit(1)
    page = pages[0]
    print("[DEBUG L205] main: got first page:", page)

    print("Encoding image and sending to model...")
    print("[DEBUG L208] main: about to encode_image")
    img_b64 = encode_image(page)
    print("[DEBUG L210] main: encode_image done")

    start = time.time()
    print("[DEBUG L213] main: set start time")
    try:
        print("[DEBUG L215] main: about to transcribe_page")
        rows = transcribe_page(img_b64)
        print("[DEBUG L217] main: transcribe_page done")
    except requests.exceptions.Timeout:
        print(f"[error] request timed out after {REQUEST_TIMEOUT}s")
        print("[DEBUG L220] main: timeout, exiting")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"[error] request failed: {e}")
        print("[DEBUG L224] main: request error, exiting")
        sys.exit(1)
    elapsed = time.time() - start
    print("[DEBUG L227] main: computed elapsed")

    gas_rows = [r for r in rows if is_gas_row(r)]
    print("[DEBUG L230] main: filtered gas_rows")

    print(f"\n{len(rows)} row(s) transcribed, {len(gas_rows)} gas row(s) found -- {elapsed:.1f}s")
    print("[DEBUG L233] main: printed summary")

    print("[DEBUG L235] main: opening february_all_rows.json")
    with open("february_all_rows.json", "w") as f:
        print("[DEBUG L237] main: writing all rows")
        json.dump(rows, f, indent=2)
        print("[DEBUG L239] main: wrote all rows")

    print("[DEBUG L241] main: opening february_gas_receipts.json")
    with open("february_gas_receipts.json", "w") as f:
        print("[DEBUG L243] main: writing gas rows")
        json.dump(gas_rows, f, indent=2)
        print("[DEBUG L245] main: wrote gas rows")

    print("[DEBUG L247] main: opening summary.txt")
    gas_total = sum(parse_amount(r.get("amount")) for r in gas_rows)
    print("[DEBUG L248] main: computed gas_total:", gas_total)
    with open("summary.txt", "w") as f:
        print("[DEBUG L249] main: writing summary header")
        f.write(f"Source PDF: {pdf_path}\n")
        print("[DEBUG L251] main: wrote source PDF line")
        f.write(f"Total rows transcribed: {len(rows)}\n")
        print("[DEBUG L253] main: wrote total rows line")
        f.write(f"Gas transactions found: {len(gas_rows)}\n")
        print("[DEBUG L255] main: wrote gas count line")
        f.write(f"Processing time: {elapsed:.1f}s\n")
        print("[DEBUG L256] main: wrote elapsed line")

        f.write("\nGas transactions:\n")
        print("[DEBUG L258] main: writing gas transaction breakdown")
        for r in gas_rows:
            date = r.get("date", "")
            merchant = r.get("merchant", "")
            amount = parse_amount(r.get("amount"))
            f.write(f"  {date} | {merchant} | ${amount:.2f}\n")
        print("[DEBUG L263] main: wrote all gas transaction lines")

        f.write(f"\nTotal gas cost: ${gas_total:.2f}\n")
        print("[DEBUG L266] main: wrote total gas cost line")

    print("Output files: february_all_rows.json, february_gas_receipts.json, summary.txt")
    print("[DEBUG L260] main: done")


if __name__ == "__main__":
    print("[DEBUG L264] calling main()")
    main()
    print("[DEBUG L266] main() returned")