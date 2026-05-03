# Standard library imports for handling command-line arguments, CSV files, JSON, OS paths, and regex.
import argparse
import csv
import json
import os
import re
# Import the time module and alias it to avoid conflicting with the datetime.time object.
import time as time_module
# Time and date handling libraries, crucial for aligning 24/7 crypto markets with 9-to-5 stock markets.
from datetime import datetime, time, timedelta, timezone
# Libraries to handle HTTP requests directly without needing third-party packages like 'requests'.
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
# Library for robust timezone support.
from zoneinfo import ZoneInfo

# Define the base URLs for Polymarket's two distinct APIs.
EVENTS_URL = "https://gamma-api.polymarket.com/events"
PRICES_HISTORY_URL = "https://clob.polymarket.com/prices-history"

# Set standard HTTP headers so the API doesn't block us as a low-effort bot.
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

# Define strict constants for time filtering: US Eastern Time, 9:30 AM open, 4:00 PM close.
EASTERN = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
# Define constraints for how much data we request from the API at once to avoid timeouts.
MAX_CHUNK_DAYS = 14
PAGE_LIMIT = 500


# Create a custom Exception class to easily identify errors related specifically to our API calls.
class ApiError(Exception):
    """Raised when a Polymarket API call fails."""


# Define the massive configuration block for the Command-Line Interface.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Polymarket daily open/close prices during US stock hours."
    )
    # The main text query to search for.
    parser.add_argument(
        "event",
        nargs="?",
        type=str,
        help="Event query for single mode (example: 'NVIDIA (NVDA) Up or Down on April 16')",
    )
    # The operating mode: single event or massive date-range backfill.
    parser.add_argument(
        "--mode",
        choices=["single", "backfill"],
        default="single",
        help="single: one query, backfill: iterate date range with daily events",
    )
    # Prefix used to automatically guess event names during backfill mode.
    parser.add_argument(
        "--asset-prefix",
        type=str,
        default="NVIDIA (NVDA) Up or Down on",
        help="Backfill event prefix (default: NVIDIA (NVDA) Up or Down on)",
    )
    # The stock ticker, used to guess the URL "slug" of the event.
    parser.add_argument(
        "--ticker-symbol",
        type=str,
        default="NVDA",
        help="Ticker symbol used for date-based event slug lookup (default: NVDA)",
    )
    # Date constraints.
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    # Which side of the bet we want (e.g., 0 might be "Yes", 1 might be "No").
    parser.add_argument(
        "--outcome-index",
        type=int,
        default=0,
        help="Outcome index from the market outcomes list (default: 0)",
    )
    # How granular the price history should be (60 seconds = 1 minute candles).
    parser.add_argument("--fidelity", type=int, default=60, help="Price-history fidelity in seconds")
    # Where to save the standard daily open/close output.
    parser.add_argument(
        "--csv",
        type=str,
        default="data/processed/polymarket_daily_open_close.csv",
        help="CSV output path",
    )
    # Pagination limits to prevent infinite loops if an event can't be found.
    parser.add_argument(
        "--max-pages-open",
        type=int,
        default=30,
        help="Search depth for open events during fallback lookup (default: 30 pages)",
    )
    parser.add_argument(
        "--max-pages-closed",
        type=int,
        default=80,
        help="Search depth for closed events during fallback lookup (default: 80 pages)",
    )
    # An optimization flag: skips slow keyword searching and just guesses the exact API URL string.
    parser.add_argument(
        "--slug-only",
        action="store_true",
        help="Resolve event by deterministic date slug only (skip expensive fallback search)",
    )
    # Feature engineering flags: trigger the creation of predictive modeling datasets.
    parser.add_argument(
        "--export-intraday",
        action="store_true",
        help="Export intraday mispricing-label rows in addition to daily open/close rows",
    )
    parser.add_argument(
        "--intraday-csv",
        type=str,
        default="data/processed/polymarket_intraday_mispricing.csv",
        help="Intraday output CSV path used when --export-intraday is set",
    )
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        default=60,
        help="Prediction horizon in minutes for intraday forward-return labels",
    )
    parser.add_argument(
        "--missing-dates-csv",
        type=str,
        default="",
        help="Optional CSV path to save missing dates and reasons from backfill mode",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-day console logging (recommended for long backfills)",
    )
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV export")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    return parser.parse_args()


# Helper function to parse 'YYYY-MM-DD' strings into Python date objects.
def parse_date(date_str):
    if not date_str:
        return None
    return datetime.strptime(date_str, "%Y-%m-%d").date()


# Helper function to safely convert values to floats, returning None instead of crashing.
def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Helper function to safely convert values to integers.
def to_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# Complex helper function to normalize different timestamp formats into standard UNIX epoch seconds.
def to_epoch(value):
    if value is None:
        return None

    # If it's already a number, figure out if it's in seconds or milliseconds.
    if isinstance(value, (int, float)):
        ts = float(value)
        # If the number is huge, it's in milliseconds. Divide by 1000 to get seconds.
        return ts / 1000.0 if ts > 10_000_000_000 else ts

    # If it's text, try to parse it.
    text = str(value).strip()
    if not text:
        return None

    # Try parsing text as a direct number first.
    numeric_value = to_float(text)
    if numeric_value is not None:
        return numeric_value / 1000.0 if numeric_value > 10_000_000_000 else numeric_value

    # Otherwise, assume it's an ISO timestamp string (e.g., "2024-04-16T15:00:00Z").
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00" # Python prefers +00:00 to Z for UTC.
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


# Safely extracts a JSON list from either a direct list object or a stringified list.
def parse_json_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


# Standardizes text by converting to lowercase and stripping out special characters.
def normalize_text(text):
    lowered = str(text or "").lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


# The core networking function to communicate with Polymarket APIs.
def http_get_json(url, params=None):
    # Safely encode URL parameters (e.g., converting spaces to %20).
    query = urlencode(params or {}, doseq=True)
    full_url = f"{url}?{query}" if query else url
    request = Request(full_url, headers=HTTP_HEADERS)

    retries = 3
    for attempt in range(1, retries + 1):
        try:
            # Attempt to open the URL with a 30-second timeout.
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            # If we get rate-limited (429) or server errors (50x), sleep and try again.
            if error.code in {429, 500, 502, 503, 504} and attempt < retries:
                time_module.sleep(attempt)
                continue
            # If it's a hard error (like 404 Not Found) or we are out of retries, crash.
            raise ApiError(f"GET {full_url} failed: {error}") from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            # Handle non-HTTP network failures similarly.
            if attempt < retries:
                time_module.sleep(attempt)
                continue
            raise ApiError(f"GET {full_url} failed: {error}") from error


# Generator function that pages through Polymarket events to find what we are looking for.
def iter_events(closed, start_date_min=None, start_date_max=None, max_pages=30):
    for page in range(max_pages):
        params = {
            "limit": PAGE_LIMIT,
            "offset": page * PAGE_LIMIT,
            "closed": "true" if closed else "false",
            "order": "id",
            "ascending": "false", # Get newest events first.
        }
        # Apply date filters if provided to narrow down the search.
        if start_date_min is not None:
            params["start_date_min"] = f"{start_date_min.isoformat()}T00:00:00Z"
        if start_date_max is not None:
            params["start_date_max"] = f"{start_date_max.isoformat()}T23:59:59Z"

        payload = http_get_json(EVENTS_URL, params=params)
        if isinstance(payload, dict):
            payload = payload.get("data", [])
        if not isinstance(payload, list) or not payload:
            break

        # Yield each event one by one (this allows the consumer to stop the loop early once it finds a match).
        for event in payload:
            yield event

        # If we received fewer events than the limit, we've hit the end of the results.
        if len(payload) < PAGE_LIMIT:
            break


# A fast way to find an event if you know its exact URL slug.
def fetch_event_by_slug(slug):
    payload = http_get_json(EVENTS_URL, params={"slug": slug})
    if isinstance(payload, list) and payload:
        return payload[0]
    return None


# Helper to find a stock ticker inside a query string (e.g., extracting "NVDA" from "NVIDIA (NVDA)").
def extract_symbol_from_text(text, fallback_symbol):
    # Look for text inside parentheses first.
    match = re.search(r"\(([A-Za-z0-9._-]+)\)", str(text or ""))
    if match:
        return match.group(1).upper()

    # Look for a standalone 2-to-6 letter uppercase word.
    tokens = re.findall(r"\b[A-Z]{2,6}\b", str(text or ""))
    if tokens:
        return tokens[0].upper()

    return str(fallback_symbol or "NVDA").upper()


# Formats the event query for backfill mode (e.g., "NVIDIA (NVDA) Up or Down on April 16").
def build_daily_event_query(asset_prefix, day):
    return f"{asset_prefix} {day.strftime('%B')} {day.day}"


# Predicts what Polymarket's URL slug will be for a given date.
def build_daily_slug(symbol, day):
    return f"{symbol.lower()}-up-or-down-on-{day.strftime('%B').lower()}-{day.day}-{day.year}"


# Grades how well a fetched event matches what the user searched for.
def event_match_score(event, query):
    query_norm = normalize_text(query)
    if not query_norm:
        return 0

    best = 0
    title_norm = normalize_text(event.get("title", ""))
    # Perfect match.
    if query_norm == title_norm:
        best = max(best, 100)
    # Partial match.
    elif query_norm in title_norm:
        best = max(best, 90)

    # Check the underlying markets (questions) within the event as well.
    for market in event.get("markets", []):
        question_norm = normalize_text(market.get("question", ""))
        if query_norm == question_norm:
            best = max(best, 100)
        elif query_norm in question_norm:
            best = max(best, 90)

    return best


# The fallback mechanism: manually searches through hundreds of events if the direct slug lookup fails.
def search_event_by_query(query, date_hint=None, max_pages_open=30, max_pages_closed=80):
    # Narrow the search window to +/- 3 days around the hint to save API calls.
    search_min = date_hint - timedelta(days=3) if date_hint else None
    search_max = date_hint + timedelta(days=3) if date_hint else None

    best_event = None
    best_key = (-1, -1)

    # Search open events first, then closed events.
    pools = ((False, max_pages_open), (True, max_pages_closed))
    for closed, page_limit in pools:
        for event in iter_events(
            closed=closed,
            start_date_min=search_min,
            start_date_max=search_max,
            max_pages=page_limit,
        ):
            score = event_match_score(event, query)
            if score <= 0:
                continue

            # Keep track of the highest scoring event.
            key = (score, to_int(event.get("id")))
            if key > best_key:
                best_key = key
                best_event = event

            # If we find a perfect 100% match, stop searching immediately to save time.
            if score >= 100:
                return event

    return best_event


# High-level orchestrator to find an event using either the fast slug method or the slow search method.
def resolve_event(query, date_hint, ticker_symbol, max_pages_open, max_pages_closed, slug_only=False):
    # Try the fast deterministic slug approach first.
    if date_hint is not None:
        symbol = extract_symbol_from_text(query, ticker_symbol)
        slug = build_daily_slug(symbol, date_hint)
        event = fetch_event_by_slug(slug)
        if event is not None:
            return event

    # If the user strictly told us not to fall back, give up.
    if slug_only:
        return None

    # Fall back to keyword searching.
    return search_event_by_query(
        query=query,
        date_hint=date_hint,
        max_pages_open=max_pages_open,
        max_pages_closed=max_pages_closed,
    )


# Events can have multiple "markets" (sub-questions). This finds the right one.
def pick_market_for_query(event, query):
    markets = event.get("markets", [])
    if not isinstance(markets, list) or not markets:
        return None

    query_norm = normalize_text(query)
    best_market = None
    best_score = -1

    # Score each market based on how closely its question matches our query.
    for market in markets:
        token_ids = parse_json_list(market.get("clobTokenIds", []))
        if not token_ids:
            continue

        score = 0
        question_norm = normalize_text(market.get("question", ""))
        if query_norm and query_norm == question_norm:
            score = 3
        elif query_norm and query_norm in question_norm:
            score = 2

        if score > best_score:
            best_score = score
            best_market = market

    return best_market


# Extracts the specific crypto Token ID used to track prices for a specific outcome ("Yes" vs "No").
def pick_outcome_token(market, outcome_index):
    token_ids = parse_json_list(market.get("clobTokenIds", []))
    outcomes = parse_json_list(market.get("outcomes", []))

    if not token_ids:
        return None, None

    # Make sure the user didn't ask for an index that doesn't exist.
    idx = outcome_index
    if idx < 0 or idx >= len(token_ids):
        idx = 0

    outcome_label = outcomes[idx] if idx < len(outcomes) else f"OUTCOME_{idx}"
    return str(token_ids[idx]), str(outcome_label)


# A UI helper that prompts the user to confirm the bot found the right event before downloading massive datasets.
def confirm_event(query, event, market, outcome_label, auto_yes=False, quiet=False):
    matched_title = event.get("title", "(no title)")
    matched_question = market.get("question", "(no question)")

    if not quiet:
        print(f"Event query: {query}")
        print(f"Matched event: {matched_title}")
        print(f"Matched market question: {matched_question}")
        print(f"Event ID: {event.get('id')} | Market ID: {market.get('id')}")
        print(f"Selected outcome: {outcome_label}")

    if auto_yes:
        return True

    try:
        answer = input("Continue with this event? [y/N]: ").strip().lower()
    except EOFError:
        return False

    return answer in {"y", "yes"}


# Fetches a broad 1-day interval history if no specific dates are provided.
def fetch_history_interval(token_id, interval="1d"):
    payload = http_get_json(PRICES_HISTORY_URL, params={"market": token_id, "interval": interval})
    history = payload.get("history", []) if isinstance(payload, dict) else []
    return history if isinstance(history, list) else []


# Fetches highly granular (e.g., minute-by-minute) price data for a specific time window.
def fetch_history_chunk(token_id, start_ts, end_ts, fidelity):
    params = {
        "market": token_id,
        "startTs": int(start_ts),
        "endTs": int(end_ts),
        "fidelity": int(fidelity),
    }
    payload = http_get_json(PRICES_HISTORY_URL, params=params)
    history = payload.get("history", []) if isinstance(payload, dict) else []
    return history if isinstance(history, list) else []


# Master function to fetch all historical prices, chunking requests if the date range is too large for the API.
def fetch_history(token_id, start_date=None, end_date=None, fidelity=60):
    if start_date is None and end_date is None:
        return fetch_history_interval(token_id, interval="1d")

    # If bounds are missing, default to a 14-day window.
    if start_date is None:
        start_date = end_date - timedelta(days=14)
    if end_date is None:
        end_date = start_date + timedelta(days=14)

    # Convert the dates to US Eastern timezone aware objects to match stock market hours.
    start_dt_et = datetime.combine(start_date, time.min, tzinfo=EASTERN)
    end_dt_et = datetime.combine(end_date, time.max, tzinfo=EASTERN)

    # Convert to UTC timestamps for the API request.
    start_ts = int(start_dt_et.astimezone(timezone.utc).timestamp())
    end_ts = int(end_dt_et.astimezone(timezone.utc).timestamp())

    all_rows = []
    # Maximum data we can request per API call (14 days in seconds).
    chunk_seconds = MAX_CHUNK_DAYS * 24 * 60 * 60

    cursor = start_ts
    # Loop and paginate through the time window.
    while cursor <= end_ts:
        chunk_end = min(cursor + chunk_seconds - 1, end_ts)
        rows = fetch_history_chunk(token_id, cursor, chunk_end, fidelity=fidelity)
        all_rows.extend(rows)
        cursor = chunk_end + 1

    # Remove duplicate timestamps just in case chunks overlapped.
    dedup = {}
    for row in all_rows:
        ts = to_epoch(row.get("t")) if isinstance(row, dict) else None
        price = to_float(row.get("p")) if isinstance(row, dict) else None
        if ts is None or price is None:
            continue
        dedup[int(ts)] = price

    # Return a clean list of sorted dictionaries.
    return [{"t": ts, "p": dedup[ts]} for ts in sorted(dedup.keys())]


# Filters the raw 24/7 price data down to just the Open and Close prices during US stock market hours.
def build_daily_open_close(history_rows, start_date=None, end_date=None):
    intraday = []

    for row in history_rows:
        if not isinstance(row, dict):
            continue

        ts = to_epoch(row.get("t"))
        price = to_float(row.get("p"))
        if ts is None or price is None:
            continue

        # Convert Polymarket UTC time to Wall Street Time.
        dt_et = datetime.fromtimestamp(ts, timezone.utc).astimezone(EASTERN)
        row_date = dt_et.date()

        # Discard data outside requested bounds.
        if start_date is not None and row_date < start_date:
            continue
        if end_date is not None and row_date > end_date:
            continue

        # Discard weekends (Saturday = 5, Sunday = 6).
        if dt_et.weekday() >= 5:
            continue

        # Discard any trading that happened before 9:30 AM or after 4:00 PM.
        local_time = dt_et.time()
        if local_time < MARKET_OPEN or local_time > MARKET_CLOSE:
            continue

        intraday.append((dt_et, price))

    intraday.sort(key=lambda item: item[0])

    daily = {}
    # Iterate through the valid intraday points.
    for dt_et, price in intraday:
        day_key = dt_et.date().isoformat()
        ts_text = dt_et.strftime("%Y-%m-%d %H:%M:%S")

        # The first time we see a day, that price is the "Open".
        if day_key not in daily:
            daily[day_key] = {
                "date": day_key,
                "open_price": price,
                "close_price": price,
                "open_time_et": ts_text,
                "close_time_et": ts_text,
            }
        # As the loop continues, we keep updating the "Close" price until the day ends.
        else:
            daily[day_key]["close_price"] = price
            daily[day_key]["close_time_et"] = ts_text

    return [daily[key] for key in sorted(daily.keys())]


# Simple console logger for the final output.
def print_daily_open_close(rows):
    for row in rows:
        line = f"{row['date']} -> OPEN: {row['open_price']:.4f} | CLOSE: {row['close_price']:.4f}"
        if row.get("event_query"):
            line += f" | QUERY: {row['event_query']}"
        print(line)


# Exports the standard daily open/close data to a CSV.
def save_csv(rows, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    headers = ["date", "open_price", "close_price", "open_time_et", "close_time_et"]
    if any("event_query" in row for row in rows):
        headers.append("event_query")
    if any("matched_event" in row for row in rows):
        headers.append("matched_event")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


# A more flexible CSV exporter used for the complex Intraday datasets.
def save_generic_csv(rows, path, fieldnames):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


# Saves a report of any days where the API failed to find an event during a backfill run.
def save_missing_dates_csv(rows, path):
    save_generic_csv(rows, path, fieldnames=["date", "reason"])


# Feature Engineering logic: Generates predictive ML labels by matching a current price 
# with what the price will be 'X' minutes in the future.
def build_intraday_mispricing_rows(
    history_rows,
    horizon_minutes,
    start_date=None,
    end_date=None,
    event_query="",
    matched_event="",
):
    points = []
    # First, filter the data down to only valid stock-market hours just like the daily function.
    for row in history_rows:
        if not isinstance(row, dict):
            continue

        ts = to_epoch(row.get("t"))
        up_price = to_float(row.get("p"))
        if ts is None or up_price is None:
            continue

        dt_et = datetime.fromtimestamp(ts, timezone.utc).astimezone(EASTERN)
        row_date = dt_et.date()

        if start_date is not None and row_date < start_date:
            continue
        if end_date is not None and row_date > end_date:
            continue
        if dt_et.weekday() >= 5:
            continue
        if dt_et.time() < MARKET_OPEN or dt_et.time() > MARKET_CLOSE:
            continue

        points.append({"dt_et": dt_et, "trade_date": row_date.isoformat(), "up_price": up_price})

    points.sort(key=lambda item: item["dt_et"])
    if not points:
        return []

    # Calculate the exact time gap we are looking for (e.g., 60 minutes).
    horizon_delta = timedelta(minutes=horizon_minutes)
    suffix = f"h{horizon_minutes}m"
    rows = []
    j = 1

    # Loop through every single minute of trading.
    for i, point in enumerate(points):
        if j < i + 1:
            j = i + 1

        # Look ahead into the future to find the target time.
        target_time = point["dt_et"] + horizon_delta
        # Fast-forward our 'j' pointer until we reach that future timestamp.
        while j < len(points) and points[j]["dt_et"] < target_time:
            j += 1

        future_price = None
        future_ts_text = ""
        future_return = None
        target_up_move = ""

        # If we successfully found a price 60 minutes in the future, and it's on the same day...
        if j < len(points) and points[j]["trade_date"] == point["trade_date"]:
            future_price = points[j]["up_price"]
            future_ts_text = points[j]["dt_et"].strftime("%Y-%m-%d %H:%M:%S")
            # Calculate the percentage change between now and the future.
            if point["up_price"] != 0:
                future_return = (future_price - point["up_price"]) / point["up_price"]
                # Create a binary ML label (1 if it went up, 0 if it went down).
                target_up_move = int(future_return > 0)

        # Compile the final row for this minute.
        row = {
            "trade_date": point["trade_date"],
            "timestamp_et": point["dt_et"].strftime("%Y-%m-%d %H:%M:%S"),
            "up_price": point["up_price"],
            "down_price": 1 - point["up_price"],
            f"future_timestamp_et_{suffix}": future_ts_text,
            f"future_up_price_{suffix}": future_price if future_price is not None else "",
            f"future_return_{suffix}": future_return if future_return is not None else "",
            f"target_up_move_{suffix}": target_up_move,
            "event_query": event_query,
            "matched_event": matched_event,
        }
        rows.append(row)

    return rows


# Executes the logic for looking up and scraping a single user-provided query.
def run_single_mode(args, start_date, end_date):
    if not args.event:
        print("Single mode requires an event query argument.")
        return 1

    date_hint = start_date or end_date
    # 1. Find the event ID.
    event = resolve_event(
        query=args.event,
        date_hint=date_hint,
        ticker_symbol=args.ticker_symbol,
        max_pages_open=args.max_pages_open,
        max_pages_closed=args.max_pages_closed,
        slug_only=args.slug_only,
    )
    if not event:
        print("No matching event found")
        return 1

    # 2. Find the market ID.
    market = pick_market_for_query(event, args.event)
    if not market:
        print("Matched event has no market with token IDs")
        return 1

    # 3. Find the exact Token ID for the Yes/No side.
    token_id, outcome_label = pick_outcome_token(market, args.outcome_index)
    if not token_id:
        print("Matched market has no token IDs")
        return 1

    # 4. Ask user to confirm.
    if not confirm_event(args.event, event, market, outcome_label, auto_yes=args.yes, quiet=args.quiet):
        print("Cancelled by user")
        return 0

    # 5. Download the raw price history.
    history_rows = fetch_history(
        token_id,
        start_date=start_date,
        end_date=end_date,
        fidelity=args.fidelity,
    )
    if not history_rows:
        print("No historical price data returned")
        return 1

    # 6. Process the daily open/close.
    daily_rows = build_daily_open_close(
        history_rows,
        start_date=start_date,
        end_date=end_date,
    )
    if not daily_rows:
        print("No valid intraday rows in stock-market hours for this range")
        return 1

    matched_title = str(event.get("title", ""))
    for row in daily_rows:
        row["event_query"] = args.event
        row["matched_event"] = matched_title

    # 7. Output results to terminal and CSV.
    if not args.quiet:
        print_daily_open_close(daily_rows)

    if not args.no_csv:
        save_csv(daily_rows, args.csv)
        print(f"Saved CSV: {args.csv}")

    # 8. Run the complex ML labeling logic if requested.
    if args.export_intraday:
        intraday_rows = build_intraday_mispricing_rows(
            history_rows,
            horizon_minutes=args.horizon_minutes,
            start_date=start_date,
            end_date=end_date,
            event_query=args.event,
            matched_event=matched_title,
        )
        suffix = f"h{args.horizon_minutes}m"
        fieldnames = [
            "trade_date",
            "timestamp_et",
            "up_price",
            "down_price",
            f"future_timestamp_et_{suffix}",
            f"future_up_price_{suffix}",
            f"future_return_{suffix}",
            f"target_up_move_{suffix}",
            "event_query",
            "matched_event",
        ]
        save_generic_csv(intraday_rows, args.intraday_csv, fieldnames=fieldnames)
        print(f"Saved intraday CSV: {args.intraday_csv}")
        print(f"Intraday rows: {len(intraday_rows)}")

    return 0


# Executes the logic for looping over a massive date range and scraping daily data automatically.
def run_backfill_mode(args, start_date, end_date):
    if start_date is None or end_date is None:
        print("Backfill mode requires both --start and --end dates")
        return 1

    rows = []
    missing_records = []
    intraday_rows_all = []
    current = start_date

    # Loop day by day until we hit the end date.
    while current <= end_date:
        # Skip weekends automatically.
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        # Dynamically build the text query (e.g., "NVIDIA Up or Down on {Current Day}").
        query = build_daily_event_query(args.asset_prefix, current)
        
        # 1. Find the event ID for this specific day.
        event = resolve_event(
            query=query,
            date_hint=current,
            ticker_symbol=args.ticker_symbol,
            max_pages_open=args.max_pages_open,
            max_pages_closed=args.max_pages_closed,
            slug_only=args.slug_only,
        )

        # Log it if we can't find a market for this day (e.g., holidays).
        if not event:
            missing_records.append({"date": current.isoformat(), "reason": "no_matching_event"})
            if not args.quiet:
                print(f"{current.isoformat()} -> no matching event")
            current += timedelta(days=1)
            continue

        # 2 & 3. Extract the right IDs.
        market = pick_market_for_query(event, query)
        if not market:
            missing_records.append(
                {"date": current.isoformat(), "reason": "matched_event_has_no_token_ids"}
            )
            if not args.quiet:
                print(f"{current.isoformat()} -> matched event has no token IDs")
            current += timedelta(days=1)
            continue

        token_id, outcome_label = pick_outcome_token(market, args.outcome_index)
        if not token_id:
            missing_records.append({"date": current.isoformat(), "reason": "unable_to_select_token"})
            if not args.quiet:
                print(f"{current.isoformat()} -> unable to select outcome token")
            current += timedelta(days=1)
            continue

        # Only confirm on the very first day. Once approved, the rest of the loop runs silently.
        if not confirm_event(
            query,
            event,
            market,
            outcome_label,
            auto_yes=args.yes,
            quiet=args.quiet,
        ):
            print("Cancelled by user")
            return 0

        # 4 & 5. Fetch and process the data just for this specific day.
        history_rows = fetch_history(
            token_id,
            start_date=current,
            end_date=current,
            fidelity=args.fidelity,
        )
        daily_rows = build_daily_open_close(
            history_rows,
            start_date=current,
            end_date=current,
        )

        if not daily_rows:
            missing_records.append(
                {"date": current.isoformat(), "reason": "no_market_hours_intraday_data"}
            )
            if not args.quiet:
                print(f"{current.isoformat()} -> no market-hours intraday data")
            current += timedelta(days=1)
            continue

        # Add this day's results to the master lists.
        day_row = daily_rows[-1]
        day_row["event_query"] = query
        matched_title = str(event.get("title", ""))
        day_row["matched_event"] = matched_title
        rows.append(day_row)

        if args.export_intraday:
            intraday_rows = build_intraday_mispricing_rows(
                history_rows,
                horizon_minutes=args.horizon_minutes,
                start_date=current,
                end_date=current,
                event_query=query,
                matched_event=matched_title,
            )
            intraday_rows_all.extend(intraday_rows)

        if not args.quiet:
            print(
                f"{day_row['date']} -> OPEN: {day_row['open_price']:.4f} | "
                f"CLOSE: {day_row['close_price']:.4f}"
            )

        # Move to the next day.
        current += timedelta(days=1)

    # Output final compiled CSVs after the entire loop is finished.
    rows.sort(key=lambda item: item["date"])

    if not rows:
        print("No rows collected in backfill mode")
        return 1

    if not args.no_csv:
        save_csv(rows, args.csv)
        print(f"Saved CSV: {args.csv}")

    if args.export_intraday:
        suffix = f"h{args.horizon_minutes}m"
        fieldnames = [
            "trade_date",
            "timestamp_et",
            "up_price",
            "down_price",
            f"future_timestamp_et_{suffix}",
            f"future_up_price_{suffix}",
            f"future_return_{suffix}",
            f"target_up_move_{suffix}",
            "event_query",
            "matched_event",
        ]
        save_generic_csv(intraday_rows_all, args.intraday_csv, fieldnames=fieldnames)
        print(f"Saved intraday CSV: {args.intraday_csv}")
        print(f"Intraday rows: {len(intraday_rows_all)}")

    if args.missing_dates_csv:
        save_missing_dates_csv(missing_records, args.missing_dates_csv)
        print(f"Saved missing-dates CSV: {args.missing_dates_csv}")

    print(f"Collected rows: {len(rows)}")
    print(f"Missing dates: {len(missing_records)}")
    if (not args.quiet) and missing_records:
        print("Missing dates: " + ", ".join(item["date"] for item in missing_records))

    return 0


# The entry point of the script.
def main():
    # Gather CLI arguments.
    args = parse_args()

    # Make sure the dates are valid.
    start_date = parse_date(args.start)
    end_date = parse_date(args.end)
    if start_date is not None and end_date is not None and start_date > end_date:
        print("Invalid date range: start is after end")
        return 1

    try:
        # Route the program to the right mode based on the user's --mode flag.
        if args.mode == "backfill":
            return run_backfill_mode(args, start_date, end_date)
        return run_single_mode(args, start_date, end_date)
        
    # Catch known errors and exit gracefully without showing a messy stack trace.
    except ApiError as error:
        print(f"Request failed: {error}")
        return 1
    except ValueError as error:
        print(f"Failed to parse response: {error}")
        return 1


# Standard boilerplate to ensure the main() function only runs if executed directly.
if __name__ == "__main__":
    raise SystemExit(main())