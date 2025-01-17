import json
import pandas as pd
from typing import Optional

def extract_output_section(analysis: str) -> Optional[str]:
    try:
        # List of possible output section markers with variations in markdown formatting
        markers = [
            "## 2. Output",
            "### Output",
            "\nOutput:",
            "**Output:**",
            "\n## Output",
            "2. Output:",
            "\n2. Output",
            "### 2. Output",
            "\n**Output",  # New pattern
            "Output"  # Most generic case
        ]
        
        # Find the last occurrence of any marker
        last_marker_pos = -1
        found_marker = None
        
        for marker in markers:
            pos = analysis.rfind(marker)
            if pos > last_marker_pos:
                last_marker_pos = pos
                found_marker = marker
                
        if last_marker_pos == -1:
            return None
            
        # Extract everything after the marker
        output_section = analysis[last_marker_pos + len(found_marker):]
        
        # Clean up the output
        output_section = output_section.strip(':').strip()
        
        # Remove any trailing sections, handling multiple newline types
        output_section = output_section.split('\n\n')[0].split('\r\n')[0].split('\n')[0]
        
        # Clean up markdown and extra whitespace
        output_section = output_section.strip('*').strip()
        
        # Handle completely bold output (text between ** **)
        if output_section.startswith('**') and '**' in output_section[2:]:
            start = 2
            end = output_section[2:].find('**') + 2
            bold_text = output_section[start:end].strip()
            if bold_text:
                output_section = bold_text
        
        # Additional cleanup for any remaining markdown or quotes
        output_section = output_section.replace('**', '').replace('*', '').strip('"').strip()
        
        # Validate that we have meaningful content
        if len(output_section) < 10 or output_section.startswith('#'):
            return None
            
        return output_section if output_section else None
    except Exception as e:
        print(f"Error in extraction: {str(e)}")
        return None

def main():
    # Create file dialog for input file
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    print("Please select the input JSONL file...")
    input_file = filedialog.askopenfilename(
        title="Select input JSONL file",
        filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
    )
    
    if not input_file:
        print("No file selected. Exiting...")
        return
        
    print(f"Selected file: {input_file}")
    
    # Read the JSONL file
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
            
    print(f"\nProcessing {len(results)} entries...")
    
    # Extract outputs and track failures
    successful_extractions = 0
    failed_markers = []
    
    for result in results:
        if 'Analysis' in result:
            output = extract_output_section(result['Analysis'])
            result['extracted_output'] = output
            if output:
                successful_extractions += 1
            else:
                # Store a small sample of failed texts
                if len(failed_markers) < 5:
                    sample = result['Analysis'][:200] + "..."  # Just store the start
                    failed_markers.append(sample)
    
    # Calculate success rate
    success_rate = (successful_extractions / len(results)) * 100
    print(f"\nSuccessfully extracted outputs for {successful_extractions} out of {len(results)} entries")
    print(f"Success rate: {success_rate:.2f}%")
    
    # Show examples
    print("\nExample extractions (first 5 successful):")
    shown = 0
    for result in results:
        if result.get('extracted_output') and shown < 5:
            print(f"\nOriginal target: {result.get('target', 'N/A')}")
            print(f"Extracted output: {result['extracted_output']}")
            shown += 1
    
    # Show some failed examples
    if failed_markers:
        print("\nSample of texts where extraction failed (first 200 chars):")
        for i, sample in enumerate(failed_markers, 1):
            print(f"\nFailed example {i}:")
            print(sample)
    
    # Save results
    output_file = input_file.replace('.jsonl', '_with_outputs.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
            
    print(f"\nResults saved to: {output_file}")
    
    # Read the JSONL file
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
            
    print(f"\nProcessing {len(results)} entries...")
    
    # Extract outputs
    successful_extractions = 0
    for result in results:
        if 'Analysis' in result:
            output = extract_output_section(result['Analysis'])
            result['extracted_output'] = output
            if output:
                successful_extractions += 1
    
    # Calculate success rate
    success_rate = (successful_extractions / len(results)) * 100
    print(f"\nSuccessfully extracted outputs for {successful_extractions} out of {len(results)} entries")
    print(f"Success rate: {success_rate:.2f}%")
    
    # Save results
    output_file = input_file.replace('.jsonl', '_with_outputs.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
            
    # Show some examples
    print("\nExample extractions (first 5 successful):")
    shown = 0
    for result in results:
        if result.get('extracted_output') and shown < 5:
            print(f"\nOriginal target: {result.get('target', 'N/A')}")
            print(f"Extracted output: {result['extracted_output']}")
            shown += 1
            
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()