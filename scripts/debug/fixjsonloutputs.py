import json
import pandas as pd
from pathlib import Path

def validate_json(record):
    """Test if a record can be properly serialized to JSON"""
    try:
        json.dumps(record)
        return True
    except (TypeError, json.JSONDecodeError):
        return False

def fix_testprompt2_output(input_path, output_path):
    """
    Fix testprompt2 outputs by converting to regular JSON and removing text field.
    """
    fixed_records = []
    errors = []
    
    print(f"Processing {input_path}")
    
    try:
        df = pd.read_json(input_path, lines=True)
        print(f"Successfully read {len(df)} records")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    for idx, row in df.iterrows():
        try:
            record = row.to_dict()
            output = record.get('testprompt2_output', '')
            
            # Remove the text field to reduce file size
            record.pop('text', None)
            
            # Handle the output
            if isinstance(output, str) and output.strip() == "The target country is not discussed in this speech.":
                record['testprompt2_output'] = {
                    "analysis_type": "no_content",
                    "message": output.strip()
                }
            elif isinstance(output, str):
                record['testprompt2_output'] = {
                    "analysis_type": "content_analysis",
                    "full_analysis": output.strip()
                }
            else:
                record['testprompt2_output'] = output
            
            # Print first record for debugging
            if idx == 0:
                print("\nFirst record structure:")
                print(json.dumps(record, indent=2))
            
            # Validate the record can be serialized
            if validate_json(record):
                fixed_records.append(record)
            else:
                errors.append((idx, "Invalid JSON structure", record))
                
        except Exception as e:
            errors.append((idx, str(e), None))
    
    print(f"Successfully processed {len(fixed_records)} records")
    if errors:
        print(f"Encountered {len(errors)} errors")
        for idx, error, rec in errors[:5]:
            print(f"Error at index {idx}: {error}")
    
    # Write fixed records as regular JSON
    try:
        # Convert Path object to string and create new path for JSON file
        output_json_path = str(output_path).replace('.jsonl', '.json')
        with open(output_json_path, 'w') as f:
            json.dump(fixed_records, f, indent=2)
        
        # Verify the output is readable
        with open(output_json_path, 'r') as f:
            test_data = json.load(f)
            print(f"Successfully verified output file with {len(test_data)} records")
        
    except Exception as e:
        print(f"Error writing or verifying output file: {e}")

def fix_testprompt3_output(input_path, output_path):
    """
    Fix testprompt3 outputs by converting to regular JSON and removing text field.
    """
    fixed_records = []
    errors = []
    
    print(f"\nProcessing {input_path}")
    
    try:
        df = pd.read_json(input_path, lines=True)
        print(f"Successfully read {len(df)} records")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    for idx, row in df.iterrows():
        try:
            record = row.to_dict()
            output = record.get('testprompt3_output', '')
            
            # Remove the text field to reduce file size
            record.pop('text', None)
            
            if isinstance(output, str) and output.strip() == "The target country is not discussed in this speech.":
                record['testprompt3_output'] = {
                    "analysis_type": "no_content",
                    "message": output.strip()
                }
            elif isinstance(output, str):
                try:
                    # Remove extra curly braces and clean up the string
                    json_str = output.replace('{{', '{').replace('}}', '}').strip()
                    parsed_json = json.loads(json_str)
                    record['testprompt3_output'] = parsed_json
                except json.JSONDecodeError:
                    record['testprompt3_output'] = {
                        "analysis_type": "parse_error",
                        "original_text": output.strip()
                    }
            else:
                record['testprompt3_output'] = output
            
            # Print first record for debugging
            if idx == 0:
                print("\nFirst record structure:")
                print(json.dumps(record, indent=2))
            
            # Validate the record can be serialized
            if validate_json(record):
                fixed_records.append(record)
            else:
                errors.append((idx, "Invalid JSON structure", record))
                
        except Exception as e:
            errors.append((idx, str(e), None))
    
    print(f"Successfully processed {len(fixed_records)} records")
    if errors:
        print(f"Encountered {len(errors)} errors")
        for idx, error, rec in errors[:5]:
            print(f"Error at index {idx}: {error}")
    
    # Write fixed records as regular JSON
    try:
        # Convert Path object to string and create new path for JSON file
        output_json_path = str(output_path).replace('.jsonl', '.json')
        with open(output_json_path, 'w') as f:
            json.dump(fixed_records, f, indent=2)
        
        # Verify the output is readable
        with open(output_json_path, 'r') as f:
            test_data = json.load(f)
            print(f"Successfully verified output file with {len(test_data)} records")
        
    except Exception as e:
        print(f"Error writing or verifying output file: {e}")

def main():
    base_dir = Path("C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/prompt_output")
    
    # Fix testprompt2 outputs with updated output filename
    prompt2_input = base_dir / "testprompt2/testprompt2_output_01.jsonl"
    prompt2_output = base_dir / "testprompt2/testprompt2_output_fixed.jsonl"
    fix_testprompt2_output(prompt2_input, prompt2_output)
    
    # Fix testprompt3 outputs with updated output filename
    prompt3_input = base_dir / "testprompt3/testprompt3_output.jsonl"
    prompt3_output = base_dir / "testprompt3/testprompt3_output_fixed.jsonl"
    fix_testprompt3_output(prompt3_input, prompt3_output)

if __name__ == "__main__":
    main()