import csv
import prodigy
from prodigy.util import set_hashes

def custom_csv_loader(file_path):
    with open(file_path, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row.get('text')
            if text:
                example = {
                    'text': text,
                    'meta': {
                        'source': row.get('source'),
                        'target_iso': row.get('target_iso')
                    }
                }
                example = set_hashes(example)
                yield example

@prodigy.recipe(
    "custom_textcat_teach",
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Name of the spaCy model to use", "positional", None, str),
    source=("Path to source data", "positional", None, str),
    label=("Comma-separated list of labels", "option", "l", str)
)
def custom_textcat_teach(dataset, spacy_model, source, label=None):
    stream = custom_csv_loader(source)
    return {
        'dataset': dataset,
        'view_id': 'classification',
        'stream': stream,
        'config': {
            'labels': label.split(',') if label else []
        }
    }
