from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
import datetime
import uuid
import json
import re

app = Flask(__name__)

def find_latest_model(models_dir='./models'):
    """Finds the latest model in the specified directory."""
    subdirs = [
        os.path.join(models_dir, d)
        for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No models found in {models_dir}")
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir

def parse_json_input(json_data):
    """Parses JSON input and extracts text from Objects array."""
    try:
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
            
        texts = []
        if 'Objects' in data and isinstance(data['Objects'], list):
            for obj in data['Objects']:
                if 'Text' in obj:
                    texts.append(obj['Text'])
        return texts
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error parsing JSON: {str(e)}")

def build_annotations(ner_results, text):
    """Builds annotations in the required format from NER results."""

    # Get word spans
    word_spans = [(match.start(), match.end()) for match in re.finditer(r'\b\w+\b', text)]
    
    # Build list of words with their positions
    words_with_positions = []
    for start, end in word_spans:
        word_text = text[start:end]
        words_with_positions.append({'start': start, 'end': end, 'text': word_text, 'labels': []})
    
    # For each entity, assign labels to overlapping words
    for entity in ner_results:
        label = entity['entity_group']
        entity_start = entity['start']
        entity_end = entity['end']
        
        # Find overlapping words
        for word in words_with_positions:
            word_start = word['start']
            word_end = word['end']
            # Check if the entity overlaps with the word
            if word_end <= entity_start:
                continue  # Word ends before entity starts
            elif word_start >= entity_end:
                break  # Word starts after entity ends
            else:
                # Assign label to the word if it doesn't have one already
                if not word['labels']:
                    word['labels'] = [label]
                else:
                    # Conflict: word already has a label
                    # Since one word can have only one label, we will keep the existing one
                    pass
    
    # Build annotations from words that have labels
    annotations = []
    for word in words_with_positions:
        if word['labels']:
            annotation = {
                "value": {
                    "start": word['start'],
                    "end": word['end'],
                    "text": word['text'],
                    "labels": word['labels']
                },
                "id": f"label_{uuid.uuid4()}",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual"
            }
            annotations.append(annotation)

    return annotations

@app.route('/inference', methods=['POST'])
def run_inference():
    try:
        # Handle both JSON file upload and direct JSON data
        if 'file' in request.files:
            file = request.files['file']
            file_contents = file.read().decode('utf-8')
            texts = parse_json_input(file_contents)
        elif request.is_json:
            texts = parse_json_input(request.get_json())
        else:
            return jsonify({"error": "No valid input provided"}), 400

        model_path = request.form.get('model_path')
        
        # Determine the model path
        if model_path is None:
            try:
                model_path = find_latest_model('./models')
            except FileNotFoundError as e:
                return jsonify({"error": str(e)}), 404
        else:
            model_path = os.path.join('./models', model_path)
            if not os.path.exists(model_path):
                return jsonify({"error": f"Model path {model_path} does not exist."}), 404

        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)

        # Initialize the NER pipeline
        nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

        # Run inference and build results
        results = []
        for idx, text in enumerate(texts):
            ner_results = nlp(text)
            annotations = build_annotations(ner_results, text)
            
            annotation_entry = {
                "completed_by": 1,
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "ground_truth": False,
                "id": int(uuid.uuid4().int & (1<<16)-1),
                "lead_time": 0.0,
                "prediction": {},
                "project": 3,
                "result": annotations,
                "result_count": len(annotations),
                "task": int(uuid.uuid4().int & (1<<16)-1),
                "unique_id": str(uuid.uuid4()),
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "was_cancelled": False
            }
            
            output_entry = {
                "id": idx + 1,
                "annotations": [annotation_entry],
                "data": {"text": text},
                "meta": {},
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "inner_id": idx + 1,
                "total_annotations": 1,
                "cancelled_annotations": 0,
                "total_predictions": 0,
                "comment_count": 0,
                "unresolved_comment_count": 0,
                "project": 3
            }
            results.append(output_entry)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)