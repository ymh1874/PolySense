import argparse
import json
from pathlib import Path

import pandas as pd


def load_json_records(input_path: Path):
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]

    raise ValueError("JSON input must be a list of records or an object with a 'data' list.")


def json_to_csv(input_path: Path, output_path: Path):
    records = load_json_records(input_path)
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Converted JSON to CSV: {output_path}")


def csv_to_json(input_path: Path, output_path: Path):
    df = pd.read_csv(input_path)
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Converted CSV to JSON: {output_path}")


def infer_output_path(input_path: Path, to_format: str) -> Path:
    suffix = ".csv" if to_format == "csv" else ".json"
    return input_path.with_suffix(suffix)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert files between JSON and CSV formats.")
    parser.add_argument("--input", required=True, help="Path to the input file.")
    parser.add_argument("--to", choices=["csv", "json"], required=True, help="Target format.")
    parser.add_argument("--output", help="Path to the output file. If omitted, inferred from input path.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else infer_output_path(input_path, args.to)

    if args.to == "csv":
        json_to_csv(input_path, output_path)
    else:
        csv_to_json(input_path, output_path)


if __name__ == "__main__":
    main()
