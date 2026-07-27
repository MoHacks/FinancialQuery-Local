"""
extract_gas.py

Reads a multi-page statement PDF (table-format transaction records),
transcribes every row on every page using a local Qwen2.5-VL model via
Ollama, then classifies each row as gas/fuel-related using a second,
lightweight local text model's judgment (no hardcoded keyword list).

Usage:
    python3 extract_gas.py february_statement.pdf

Requirements:
    pip install pdf2image requests tqdm --break-system-packages
    brew install poppler
    ollama pull qwen2.5vl:7b   # vision model, reads each statement page
    ollama pull qwen2.5:7b     # text model, classifies each row
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
from datetime import datetime
print("[DEBUG L27] imported datetime")

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


# Common date formats seen on statement exports. Tried in order until one
# matches; unparseable dates fall back to sorting last rather than
# crashing the sort.
DATE_FORMATS = [
    "%b %d",        # Jan 15
    "%B %d",        # January 15
    "%b %d, %Y",    # Jan 15, 2026
    "%B %d, %Y",    # January 15, 2026
    "%Y-%m-%d",      # 2026-01-15
    "%m/%d/%Y",      # 01/15/2026
    "%m/%d/%y",      # 01/15/26
    "%d/%m/%Y",      # 15/01/2026
]
print("[DEBUG L268] set DATE_FORMATS")


def parse_date_for_sort(date_str) -> datetime:
    """
    Best-effort parse of a date string into a datetime for sorting.
    Falls back to datetime.max (sorts last) if no known format matches,
    so a malformed/unusual date doesn't crash the sort -- it just ends
    up at the bottom of its page's group, easy to spot for manual review.
    """
    print("[DEBUG L276] parse_date_for_sort: enter, date_str =", date_str)
    if not date_str:
        print("[DEBUG L278] parse_date_for_sort: empty date_str, returning datetime.max")
        return datetime.max
    for fmt in DATE_FORMATS:
        try:
            result = datetime.strptime(str(date_str).strip(), fmt)
            print(f"[DEBUG L283] parse_date_for_sort: matched format '{fmt}' -> {result}")
            return result
        except ValueError:
            continue
    print("[DEBUG L287] parse_date_for_sort: no format matched, returning datetime.max")
    return datetime.max


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

    print(f"Converting {pdf_path} to images (dpi={PDF_DPI})...")
    print("[DEBUG L280] main: about to convert_from_path")
    pages = convert_from_path(pdf_path, dpi=PDF_DPI)
    print("[DEBUG L282] main: convert_from_path done")
    if not pages:
        print("No pages found in PDF.")
        print("[DEBUG L285] main: no pages, exiting")
        sys.exit(1)
    print(f"[DEBUG L287] main: got {len(pages)} page(s)")
    print(f"{len(pages)} page(s) found.\n")

    # -------------------------------------------------------------------
    # Stage 1: transcribe every page
    # -------------------------------------------------------------------

    all_rows = []
    transcribe_start = time.time()
    print("[DEBUG L296] main: starting page transcription loop")

    for i, page in enumerate(tqdm(pages, desc="Transcribing pages", unit="page")):
        print(f"[DEBUG L300] main: processing page {i + 1}")
        img_b64 = encode_image(page)
        print(f"[DEBUG L302] main: encoded page {i + 1}")

        try:
            print(f"[DEBUG L305] main: about to transcribe_page for page {i + 1}")
            rows = transcribe_page(img_b64)
            print(f"[DEBUG L307] main: transcribe_page done for page {i + 1}, {len(rows)} row(s)")
        except requests.exceptions.Timeout:
            tqdm.write(f"[error] page {i + 1} timed out after {REQUEST_TIMEOUT}s, skipping")
            print(f"[DEBUG L310] main: page {i + 1} timeout, skipping")
            continue
        except requests.exceptions.RequestException as e:
            tqdm.write(f"[error] page {i + 1} request failed: {e}, skipping")
            print(f"[DEBUG L313] main: page {i + 1} request error, skipping")
            continue

        for row in rows:
            row["_page"] = i
        all_rows.extend(rows)
        tqdm.write(f"  page {i + 1}: {len(rows)} row(s) transcribed")

    transcribe_elapsed = time.time() - transcribe_start
    print("[DEBUG L322] main: computed transcribe_elapsed")
    print(f"\n{len(all_rows)} row(s) transcribed across {len(pages)} page(s) in {transcribe_elapsed:.1f}s")

    # -------------------------------------------------------------------
    # Stage 2: classify every row across all pages
    # -------------------------------------------------------------------

    print("Classifying each row with the local text model...")
    classify_start = time.time()
    for row in tqdm(all_rows, desc="Classifying rows", unit="row"):
        row["confidence"] = classify_row(row)
    classify_elapsed = time.time() - classify_start
    print(f"[DEBUG L332] main: classified {len(all_rows)} row(s) in {classify_elapsed:.1f}s")

    gas_rows = [r for r in all_rows if r["confidence"] >= GAS_CONFIDENCE_THRESHOLD]
    gas_rows.sort(key=lambda r: (r.get("_page", 0), parse_date_for_sort(r.get("date", ""))))
    print("[DEBUG L336] main: filtered and sorted gas_rows by page then date")

    total_elapsed = transcribe_elapsed + classify_elapsed
    print(f"\n{len(gas_rows)} gas row(s) found (threshold={GAS_CONFIDENCE_THRESHOLD}) -- total {total_elapsed:.1f}s")
    print("[DEBUG L340] main: printed summary")

    # -------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------

    print(f"[DEBUG L346] main: opening {pdf_path.split('/')[-1].split('.')[0]}_all_rows.json")
    with open(f"{pdf_path.split('/')[-1].split('.')[0]}_all_rows.json", "w") as f:
        print("[DEBUG L348] main: writing all rows")
        json.dump(all_rows, f, indent=2)
        print("[DEBUG L350] main: wrote all rows")

    print(f"[DEBUG L352] main: opening {pdf_path.split('/')[-1].split('.')[0]}_gas_receipts.json")
    with open(f"{pdf_path.split('/')[-1].split('.')[0]}_gas_receipts.json", "w") as f:
        print("[DEBUG L354] main: writing gas rows")
        json.dump(gas_rows, f, indent=2)
        print("[DEBUG L356] main: wrote gas rows")

    print(f"[DEBUG L358] main: opening summary_{pdf_path.split('/')[-1].split('.')[0]}.txt")
    gas_total = sum(parse_amount(r.get("amount")) for r in gas_rows)
    print("[DEBUG L360] main: computed gas_total:", gas_total)
    with open(f"summary_{pdf_path.split('/')[-1].split('.')[0]}.txt", "w") as f:
        print("[DEBUG L362] main: writing summary header")
        f.write(f"Source PDF: {pdf_path}\n")
        f.write(f"Pages processed: {len(pages)}\n")
        f.write(f"Total rows transcribed: {len(all_rows)}\n")
        f.write(f"Gas transactions found (confidence >= {GAS_CONFIDENCE_THRESHOLD * 100:.0f}%): {len(gas_rows)}\n")
        f.write(f"Transcription time: {transcribe_elapsed:.1f}s\n")
        f.write(f"Classification time: {classify_elapsed:.1f}s\n")
        print("[DEBUG L369] main: wrote header lines")

        f.write("\nGas transactions (ordered by page, then date):\n\n")
        print("[DEBUG L372] main: writing gas transaction breakdown with per-page subtotals")

        DOTTED_LINE = "-" * 60
        current_page = None
        page_subtotal = 0.0
        page_count = 0

        for r in gas_rows:
            page_num = r.get("_page", "?")

            if current_page is not None and page_num != current_page:
                f.write(f"{DOTTED_LINE}\n")
                f.write(f"  Page {current_page + 1} gas entr{'y' if page_count == 1 else 'ies'}: {page_count}\n")
                f.write(f"  Page {current_page + 1} subtotal: ${page_subtotal:.2f}\n")
                f.write(f"{DOTTED_LINE}\n\n")
                print(f"[DEBUG L385] main: wrote subtotal for page {current_page}: {page_count} entries, ${page_subtotal:.2f}")
                page_subtotal = 0.0
                page_count = 0

            current_page = page_num
            date = r.get("date", "")
            merchant = r.get("merchant", "")
            amount = parse_amount(r.get("amount"))
            confidence = r.get("confidence", 0.0)
            f.write(f"  [page {page_num + 1}] {date} | {merchant} | ${amount:.2f} | confidence: {confidence * 100:.0f}%\n")
            page_subtotal += amount
            page_count += 1

        if current_page is not None:
            f.write(f"{DOTTED_LINE}\n")
            f.write(f"  Page {current_page + 1} gas entr{'y' if page_count == 1 else 'ies'}: {page_count}\n")
            f.write(f"  Page {current_page + 1} subtotal: ${page_subtotal:.2f}\n")
            f.write(f"{DOTTED_LINE}\n\n")
            
        print("[DEBUG L380] main: wrote all gas transaction lines")

        f.write(f"\nTotal gas cost: ${gas_total:.2f}\n")
        print("[DEBUG L383] main: wrote total gas cost line")

    print(f"Output files: {pdf_path.split('/')[-1].split('.')[0]}_all_rows.json, {pdf_path.split('/')[-1].split('.')[0]}_gas_receipts.json, summary_{pdf_path.split('/')[-1].split('.')[0]}.txt")
    print("[DEBUG L387] main: done")


if __name__ == "__main__":
    print("[DEBUG L391] calling main()")
    main()
    print("[DEBUG L393] main() returned")
