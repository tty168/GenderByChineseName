import json
import os

def infer_complete_schema(file_path):
    """
    Pass 1: Performs a global scan over the file to map out every available top-level 
    field name, valid data types, and counts the total rows.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the dataset at: {file_path}")

    schema = {}
    etymology_subkeys = set()
    records_scanned = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            cleaned_line = line.strip()
            if not cleaned_line:
                continue
            records_scanned += 1
            try:
                record = json.loads(cleaned_line)
                
                # Check top-level elements (excluding original 'etymology' dictionary)
                for key, value in record.items():
                    if key == "etymology":
                        if isinstance(value, dict):
                            etymology_subkeys.update(value.keys())
                        continue
                        
                    val_type = "null" if value is None else type(value).__name__
                    if key not in schema:
                        schema[key] = {val_type}
                    else:
                        schema[key].add(val_type)
            except json.JSONDecodeError:
                continue

    # Flatten the scanned etymology keys into the master schema tracker
    flat_etymology_fields = sorted(list(etymology_subkeys))
    for subkey in flat_etymology_fields:
        schema[f"etymology_{subkey}"] = {"str", "null"}

    # Clean sets to sorted lists for printing output
    cleaned_schema = {k: sorted(list(v)) for k, v in schema.items()}
    return cleaned_schema, flat_etymology_fields, records_scanned

def parse_and_flatten_records(file_path, flat_etymology_fields):
    """
    Pass 2: Transforms and maps the entries using the derived etymology keys.
    """
    flattened_records = []
    records_parsed = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            cleaned_line = line.strip()
            if not cleaned_line:
                continue
            try:
                record = json.loads(cleaned_line)
                etymology_data = record.pop("etymology", None) or {}
                
                # Populate structural padding
                for subkey in flat_etymology_fields:
                    record[f"etymology_{subkey}"] = etymology_data.get(subkey, None)
                    
                flattened_records.append(record)
                records_parsed += 1
            except json.JSONDecodeError:
                continue
    return flattened_records, records_parsed


import csv

def save_records_to_sqlite_csv(records, output_file_path, schema_keys):
    """
    Saves the flattened data records into a structured CSV file
    properly configured for a flawless UTF-8 SQLite import.
    """
    print(f"🔄 Exporting {len(records):,} records to CSV: '{output_file_path}'...")
    
    # 1. Ensure the schema keys are sorted to guarantee stable column positions
    column_headers = sorted(list(schema_keys))
    
    # 2. Open the file with explicit UTF-8 configurations
    with open(output_file_path, 'w', encoding='utf-8', newline='') as csvfile:
        # csv.QUOTE_MINIMAL automatically quotes fields that contain commas or line breaks
        writer = csv.DictWriter(csvfile, fieldnames=column_headers, quoting=csv.QUOTE_MINIMAL)
        
        # Write column names as the very first row
        writer.writeheader()
        
        # Write individual data rows
        for record in records:
            # Safe transformation: Convert lists (pinyin/matches) to comma-separated strings
            # This makes SQL queries much cleaner down the road
            clean_record = {}
            for key in column_headers:
                val = record.get(key, None)
                if isinstance(val, list):
                    clean_record[key] = ",".join(map(str, val))
                elif val is None:
                    clean_record[key] = ""  # SQLite imports empty strings as NULL in csv mode
                else:
                    clean_record[key] = str(val)
                    
            writer.writerow(clean_record)
            
    print("✅ CSV export complete. File is ready for SQLite CLI import.")

# --- Execution Block ---
if __name__ == "__main__":
    hanzi_file = "makemeahanzi/dictionary.txt"
    
    try:
        # 1. Discover the layout schema across the file
        master_schema, etymology_keys, total_scanned = infer_complete_schema(hanzi_file)
        
        # 2. Print Records Read in Pass 1 and Detected Schema
        print(f"🔄 Pass 1 Complete: Read {total_scanned:,} total records from file.")
        print("\n--- Target Dataset Inferred Schema ---")
        #print(json.dumps(master_schema, indent=4))
        json_str = "{\n" + ",\n".join(f'    "{k}": {json.dumps(v)}' for k, v in master_schema.items()) + "\n}"
        print(json_str)
        print("-" * 38 + "\n")
        
        # 3. Parse out the entries using schema keys
        hanzi_dataset, total_parsed = parse_and_flatten_records(hanzi_file, etymology_keys)
        print(f"🔄 Pass 2 Complete: Successfully transformed {total_parsed:,} records.\n")


        # 4. Export to CSV using the master schema keys discovered in Pass 1
        csv_output_file = "hanzi_dictionary.csv"
        save_records_to_sqlite_csv(hanzi_dataset, csv_output_file, master_schema.keys())
        
        print(f"\n🚀 Saved the hanzi dataset to CSV {csv_output_file}:")
        print(f"sqlite3 hanzi.db -cmd \".mode csv\" \".import {csv_output_file} characters\"")
        
        # 5. Print Data Samples Last
        print("--- Structural Sample Data Output ---")
        sample_chars = ["⺀", "⺊", "㐌", "湖", "想", "国"]
        for item in hanzi_dataset:
            if item.get("character") in sample_chars:
                print(f"\nGlyph: {item['character']} | Radical: {item['radical']}")
                print(f"  └─ Components:  {item.get('decomposition')}")
                print(f"  └─ Definition:  {item.get('definition')}")
                print(f"  └─ Flat Etymology Type:      {item.get('etymology_type')}")
                print(f"  └─ Flat Etymology Hint:      {item.get('etymology_hint')}")
                print(f"  └─ Flat Etymology Phonetic:  {item.get('etymology_phonetic')}")
                print(f"  └─ Flat Etymology Semantic:  {item.get('etymology_semantic')}")
                
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
