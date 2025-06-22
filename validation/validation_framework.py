from sentence_transformers import SentenceTransformer, util
import re
import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import json


model_st = SentenceTransformer('all-MiniLM-L6-v2')

def flatten_json(y, prefix=''):
    out = {}
    if isinstance(y, dict):
        for k, v in y.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(flatten_json(v, key))
    elif isinstance(y, list):
        for i, v in enumerate(y):
            key = f"{prefix}[{i}]"
            out.update(flatten_json(v, key))
    else:
        out[prefix] = y if y is not None else ''
    return out

def normalize_value(value):
    if isinstance(value, str):
        return re.sub(r'\s+', '', value.lower())
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)

def semantic_match(val1, val2, threshold=0.75):
    if not val1 or not val2:
        return False
    embedding1 = model_st.encode(str(val1), convert_to_tensor=True)
    embedding2 = model_st.encode(str(val2), convert_to_tensor=True)
    sim_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    return sim_score >= threshold

def compare_outputs(expected_json, predicted_json):
    expected = flatten_json(expected_json)
    predicted = flatten_json(predicted_json)

    all_keys = set(expected.keys()).union(predicted.keys())
    match = 0
    missing = 0
    incorrect = 0

    y_true = []
    y_pred = []

    for key in all_keys:
        expected_value = normalize_value(expected.get(key, ''))
        predicted_value = normalize_value(predicted.get(key, ''))

        if expected_value == '' and predicted_value == '':
            match += 1
            y_true.append(1)
            y_pred.append(1)
            continue

        if semantic_match(expected_value, predicted_value):
            match += 1
            y_true.append(1)
            y_pred.append(1)
        elif expected_value != '' and predicted_value != '':
            incorrect += 1
            y_true.append(1)
            y_pred.append(0)
        else:
            missing += 1
            y_true.append(1)
            y_pred.append(0)

    total = match + incorrect + missing
    accuracy = match / total if total > 0 else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'total_fields': total,
        'matched': match,
        'incorrect': incorrect,
        'missing': missing,
        'accuracy': round(accuracy, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1_score': round(f1, 3)
    }

def validate_all_models(file_path, expected_output):
    from models.mistral_ocr_handler import extract_from_mistral
    from models.azure_handler import extract_from_azure
    from models.datalab_handler import extract_from_datalab
    from models.anthropic_handler import extract_from_claude
    from models.gemini_handler import extract_from_gemini
    from models.azure_fine_tuned_handler import extract_from_azure_finetuned

    models = {
        'mistral': extract_from_mistral,
        'azure': extract_from_azure,
        'datalab': extract_from_datalab,
        'anthropic': extract_from_claude,
        'gemini': extract_from_gemini,
        'azure_finetuned': extract_from_azure_finetuned  
        
    }

    results = []
    for model_name, func in models.items():
        try:
            print(f"[DEBUG] Running model: {model_name}")
            start_time = time.time()
            output = func(file_path)
            end_time = time.time()

            if isinstance(output, str):
                output = json.loads(output)

            predicted_json = output.get("result") if isinstance(output, dict) and output.get("result") else output

            if isinstance(predicted_json, str):
                predicted_json = predicted_json.strip().strip("```json").strip("```")
                predicted_json = json.loads(predicted_json)

            metrics = compare_outputs(expected_output, predicted_json)
            metrics['model'] = model_name
            metrics['response_time_sec'] = round(end_time - start_time, 2)
            results.append(metrics)

        except Exception as e:
            results.append({
                'model': model_name,
                'total_fields': 'error',
                'matched': 'error',
                'incorrect': 'error',
                'missing': 'error',
                'accuracy': 'error',
                'precision': 'error',
                'recall': 'error',
                'f1_score': 'error',
                'response_time_sec': 'error'
            })
            print(f"[ERROR] {model_name} failed: {e}")

    return pd.DataFrame(results)
