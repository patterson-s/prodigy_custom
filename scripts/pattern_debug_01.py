import spacy
from spacy.pipeline import EntityRuler
import json

# Load spaCy model and add the EntityRuler
nlp = spacy.load("en_core_web_lg")
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Load patterns from your patterns file
patterns_path = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/patterns/institution_pattern_01.jsonl"
with open(patterns_path, "r", encoding="utf-8") as f:
    patterns = [json.loads(line) for line in f]
ruler.add_patterns(patterns)

# Test a sample text
doc = nlp("The World Health Organization announced new health guidelines.")

# Print out entities
for ent in doc.ents:
    print(ent.text, ent.label_)
