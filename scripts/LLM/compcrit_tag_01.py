import os
import cohere
import time
from pathlib import Path

def analyze_sentiment_expressions(co_client, source, target, year, summaries):
    try:
        print("\nPreparing analysis...")
        
        prompt = (
            f"You are an expert in analyzing diplomatic speech and international relations. "
            f"Your task is to identify explicit compliments or criticisms in diplomatic speech. "
            f"You will receive summaries from a UN General Assembly speech.\n\n"
            f"Analyze these summaries of how {source} discusses {target} in their {year} UN General Assembly speech:\n"
            f"{summaries}\n\n"
            f"If you find explicit compliments or criticisms, provide:\n"
            f"1. The tag [compliment_criticize]\n"
            f"2. Either \"compliment\" or \"criticism\" as appropriate, introduced by \"descriptive: \"\n\n"
            f"Example formats:\n"
            f"For \"We commend their dedication to peace...\":\n"
            f"tag: [compliment_criticize]; descriptive: \"compliment\"\n\n"
            f"For \"Their actions threaten regional stability...\":\n"
            f"tag: [compliment_criticize]; descriptive: \"criticism\"\n\n"
            f"For multiple expressions:\n"
            f"tag: [compliment_criticize]; descriptive: \"criticism\"\n"
            f"tag: [compliment_criticize]; descriptive: \"compliment\"\n\n"
            f"Only include clear expressions of praise or criticism. Look for:\n"
            f"- Direct praise or condemnation\n"
            f"- Explicit approval or disapproval\n"
            f"- Clear positive or negative judgments\n"
            f"- Expressions of gratitude or disappointment\n\n"
            f"If no clear compliments or criticisms are present, respond with NA.\n\n"
            f"Notes:\n"
            f"- Only mark clear expressions of praise or criticism\n"
            f"- Do not infer sentiment that isn't explicitly stated\n"
            f"- Each distinct compliment or criticism should be listed separately\n"
            f"- Factual statements without clear approval/disapproval should not be tagged"
        )
        
        print("Sending request to Cohere...")
        
        response = co_client.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
            k=0,
            p=0.75,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop_sequences=[],
            return_likelihoods="NONE"
        )
        
        print("Response received.")
        
        analysis = response.generations[0].text.strip()
        print("\nAnalysis Result:")
        print("-" * 50)
        print(analysis)
        print("-" * 50 + "\n")
        
        return analysis
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Initialize Cohere client
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")
    co_client = cohere.Client(api_key)

    print("Welcome to the Diplomatic Speech Compliment/Criticism Analysis Demo!")
    print("(Type 'exit' at any time to quit)")
    print("Enter speech summaries and type END (no quotes) on a new line when finished.\n")

    while True:
        try:
            source = input("Enter source country: ").strip()
            if source.lower() == "exit":
                print("Goodbye!")
                break
                
            target = input("Enter target country: ").strip()
            if target.lower() == "exit":
                print("Goodbye!")
                break
                
            year = input("Enter year: ").strip()
            if year.lower() == "exit":
                print("Goodbye!")
                break
                
            print("Enter speech summaries (type END on a new line when finished):")
            summaries = []
            while True:
                line = input().strip()
                if line.upper() == 'END':
                    break
                if line.lower() == 'exit':
                    print("Goodbye!")
                    exit()
                summaries.append(line)
            
            if not summaries:
                print("No summaries entered. Please try again.")
                continue
                
            summaries = " ".join(summaries)
            print("\nProcessing your input...")
            analysis_result = analyze_sentiment_expressions(co_client, source, target, year, summaries)
            
            time.sleep(1)
            
            retry = input("\nWould you like to try another analysis? (yes/no): ").strip().lower()
            if retry != "yes":
                print("Goodbye!")
                break
                
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            retry = input("\nWould you like to try again? (yes/no): ").strip().lower()
            if retry != "yes":
                print("Goodbye!")
                break