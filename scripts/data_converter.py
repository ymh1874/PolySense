import argparse
import json
from pathlib import Path

import pandas as pd


# Define a helper function to safely extract the actual data records from a JSON file.
def load_json_records(input_path: Path):
    # Open the file in read mode ('r') using UTF-8 encoding to handle special characters.
    with input_path.open("r", encoding="utf-8") as f:
        # Parse the JSON text into a Python object (like a list or dictionary).
        payload = json.load(f)

    # Check if the parsed JSON is already a flat list of records (e.g., [ {"id": 1}, {"id": 2} ]).
    if isinstance(payload, list):
        # If it is, just return it directly.
        return payload

    # If it's not a list, check if it's a dictionary that contains a specific key named "data" 
    # which holds the list of records (e.g., {"metadata": {...}, "data": [ {"id": 1} ]} ).
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        # If so, extract and return just that list.
        return payload["data"]

    # If the JSON doesn't match either of those expected structures, crash and tell the user why.
    raise ValueError("JSON input must be a list of records or an object with a 'data' list.")


# Define the function that converts a JSON file into a CSV file.
def json_to_csv(input_path: Path, output_path: Path):
    # Call our helper function above to grab the list of records from the JSON file.
    records = load_json_records(input_path)
    # Load those records into a pandas DataFrame. Pandas automatically aligns the dictionary keys into CSV columns.
    df = pd.DataFrame(records)
    
    # Check if the folder where we want to save the output actually exists. 
    # If it doesn't (parents=True), create it (exist_ok=True prevents an error if it already exists).
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the DataFrame to the destination path as a CSV. 
    # index=False prevents pandas from adding an extra column with the row numbers (0, 1, 2...).
    df.to_csv(output_path, index=False)
    # Print a success message to the terminal so the user knows it worked.
    print(f"Converted JSON to CSV: {output_path}")


# Define the function that converts a CSV file into a JSON file.
def csv_to_json(input_path: Path, output_path: Path):
    # Use pandas to read the CSV file directly into a DataFrame.
    df = pd.read_csv(input_path)
    
    # Convert the DataFrame back into a list of dictionaries (orient="records").
    # The 'where(pd.notnull(df), None)' part is crucial: it replaces pandas' 'NaN' (Not a Number) values 
    # with Python's 'None', which properly translates to 'null' in standard JSON.
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    
    # Ensure the destination folder exists, creating it if necessary.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open the destination file in write mode ('w').
    with output_path.open("w", encoding="utf-8") as f:
        # Write the records list into the file as JSON text.
        # indent=2 makes the JSON pretty and readable by adding line breaks and 2 spaces of indentation.
        # ensure_ascii=False ensures things like emojis or accented characters aren't scrambled into unicode strings.
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Print a success message to the terminal.
    print(f"Converted CSV to JSON: {output_path}")


# Define a helper function to guess the output filename if the user didn't provide one.
def infer_output_path(input_path: Path, to_format: str) -> Path:
    # Set the new file extension based on the target format the user requested.
    suffix = ".csv" if to_format == "csv" else ".json"
    # Take the original file path and replace its extension (e.g., 'data.json' becomes 'data.csv').
    return input_path.with_suffix(suffix)


# Define the function that configures the command-line interface.
def parse_args():
    # Create the argument parser and give it a helpful description.
    parser = argparse.ArgumentParser(description="Convert files between JSON and CSV formats.")
    # Add a required flag '--input' so the user can specify the file they want to convert.
    parser.add_argument("--input", required=True, help="Path to the input file.")
    # Add a required flag '--to' and force the user to pick either "csv" or "json".
    parser.add_argument("--to", choices=["csv", "json"], required=True, help="Target format.")
    # Add an optional flag '--output' in case they want to save it with a specific name or in a different folder.
    parser.add_argument("--output", help="Path to the output file. If omitted, inferred from input path.")
    # Parse the commands typed into the terminal and return them as a neat object.
    return parser.parse_args()


# Define the main logic block that orchestrates the script.
def main():
    # Call the parser to get the user's terminal commands.
    args = parse_args()
    # Convert the user's input string into a proper Path object.
    input_path = Path(args.input)

    # Check if the file they are trying to convert actually exists on the hard drive.
    if not input_path.exists():
        # If not, crash immediately and let them know.
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine where we are saving the file. Use the user's --output if they provided one.
    # If they didn't, call the infer_output_path helper to figure it out automatically.
    output_path = Path(args.output) if args.output else infer_output_path(input_path, args.to)

    # Check which direction we are converting based on the '--to' argument.
    if args.to == "csv":
        # Run the JSON-to-CSV logic.
        json_to_csv(input_path, output_path)
    else:
        # Run the CSV-to-JSON logic.
        csv_to_json(input_path, output_path)


# This is standard Python boilerplate. It checks if this script is being run directly 
# from the terminal (as opposed to being imported as a module into another script).
if __name__ == "__main__":
    # If it is being run directly, execute the main function.
    main()