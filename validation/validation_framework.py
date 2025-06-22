import re
import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import json

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
        return re.sub(r'\s+', '', value.lower())  # Remove ALL whitespace and lower
    if isinstance(value, (int, float)):
        return value  # Don't round unless absolutely necessary
    return value

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

        # Both empty → consider as a match (both missing is still "correct")
        if expected_value == '' and predicted_value == '':
            match += 1
            y_true.append(1)
            y_pred.append(1)
            continue

        if expected_value == predicted_value:
            match += 1
            y_true.append(1)
            y_pred.append(1)
        elif expected_value != '' and predicted_value != '':
            incorrect += 1
            y_true.append(1)
            y_pred.append(0)
        else:
            # One is missing
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

    models = {
        'mistral': extract_from_mistral,
        'azure': extract_from_azure,
        'datalab': extract_from_datalab,
        'anthropic': extract_from_claude,
        'gemini': extract_from_gemini
    }

    results = []
    for model_name, func in models.items():
        try:
            print(f"[DEBUG] Running model: {model_name}")
            start_time = time.time()
            output = func(file_path)
            end_time = time.time()
            
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    print(f"[ERROR] Model {model_name} returned invalid JSON string.")
                    raise

            predicted_json = output.get("result") if output.get("result") else output

            if isinstance(predicted_json, str):
                raw_result = predicted_json.strip()
                if raw_result.startswith('```json'):
                    raw_result = raw_result[7:]
                if raw_result.endswith('```'):
                    raw_result = raw_result[:-3]
                predicted_json = json.loads(raw_result)


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

    df = pd.DataFrame(results)
    return df
