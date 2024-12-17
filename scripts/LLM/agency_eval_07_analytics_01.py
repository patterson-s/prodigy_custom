import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Function to load results from a JSONL file
def load_results(file_path):
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            results.append(json.loads(line))
    return results

# Function to analyze adequacy rates and top-performing prompts
def analyze_results(results):
    prompt_names = []
    data = []

    # Extract relevant data
    for result in results:
        row = {"Row": result["Row"], "Preferred": result["Preferred Prompt"]}
        for prompt, adequacy in result["Adequacy"].items():
            row[prompt] = adequacy
            if prompt not in prompt_names:
                prompt_names.append(prompt)
        data.append(row)

    df = pd.DataFrame(data)
    print("\n### Adequacy Summary ###")

    # Adequacy rates for each prompt
    adequacy_summary = df[prompt_names].mean() * 100
    print(adequacy_summary)
    print("\nTop Performing Prompt:", adequacy_summary.idxmax(), "with", round(adequacy_summary.max(), 2), "% adequacy.")

    # Analyze preferred prompts
    preferred_counts = df["Preferred"].value_counts()
    print("\n### Preferred Prompt Distribution ###")
    print(preferred_counts)

    return adequacy_summary, preferred_counts

# Function to plot the results
def plot_results(adequacy_summary, preferred_counts):
    # Adequacy Bar Plot
    plt.figure(figsize=(8, 5))
    adequacy_summary.sort_values().plot(kind='bar')
    plt.title("Adequacy Rate by Prompt")
    plt.ylabel("Adequacy Rate (%)")
    plt.xlabel("Prompt")
    plt.tight_layout()
    plt.show()

    # Preferred Prompt Pie Chart
    plt.figure(figsize=(6, 6))
    preferred_counts.plot(kind='pie', autopct="%1.1f%%")
    plt.title("Preferred Prompt Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# Main function to run the analysis
def main():
    file_path = input("Enter the path to the intermediate_results.jsonl file: ").strip()
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    results = load_results(file_path)
    adequacy_summary, preferred_counts = analyze_results(results)
    plot_results(adequacy_summary, preferred_counts)

if __name__ == "__main__":
    main()
