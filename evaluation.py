import json
import requests


def classification_metrics(true_labels, precision_labels):
    """
    Calculates Accuracy, Precision, Recall, and F1 for a binary classification task.
    Returns a dict with these four metrics.
    """
    tp = fp = fn = tn = 0
    for t, p in zip(true_labels, precision_labels):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def precision_at_k(retrieved_ids, relevant_ids, k=5):
    """
    retrieved_ids: list of product IDs in ranked order (top five)
    relevant_ids: list of ground-truth relevant product IDs
    k: cutoff (top-k)
    """
    relevant_set = set(relevant_ids)
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for rid in retrieved_k if rid in relevant_set)
    return hits / k


def average_precision(retrieved_ids, relevant_ids):
    """
    Computes Average Precision (AP) for a single query.
    For each position i (1-based) in the retrieved_ids list:
      - Add Precision@i if the item at position i is relevant.
      - Then divide by the total number of relevant items.
    """
    relevant_set = set(relevant_ids)
    hits = 0
    score = 0.0
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            hits += 1
            score += hits / i
    if len(relevant_set) == 0:
        return 0.0
    return score / len(relevant_set)


def evaluate_queries(queries, top_k=5):
    precision_list = []
    ap_list = []
    retrieval_map = {}

    all_true_labels = []
    all_prediction_labels = []

    for q in queries:
        payload = {
            "query": q["query"],
            "language": q["language"],
            "sentiment": q["sentiment"],
            "model": q["model"]
        }
        response = requests.post("http://localhost:5000/search", json=payload)
        if response.status_code == 200:
            results = response.json()

            # Extract product IDs
            retrieved_ids = [item["product_id"] for item in results]
            relevant_ids = q.get("relevant_products", [])

            # Calculate Precision@k
            p_k = precision_at_k(retrieved_ids, relevant_ids, k=top_k)
            precision_list.append(p_k)

            # Calculate Average Precision
            ap = average_precision(retrieved_ids, relevant_ids)
            ap_list.append(ap)

            # Store the top_k items for overlap
            key = (q["query"], q["language"], q["sentiment"])
            retrieval_map.setdefault(key, {})[q["model"]] = retrieved_ids[:top_k]

            # Store classification labels
            for item in results:
                all_true_labels.append(item["true_label"])
                all_prediction_labels.append(item["pred_label"])
        else:
            print(f"[ERROR] Query failed ({response.status_code}): {payload}")

    # Aggregate results
    mean_precision = sum(precision_list) / len(precision_list) if precision_list else 0.0
    mean_ap = sum(ap_list) / len(ap_list) if ap_list else 0.0
    acc = precision = rec = f1 = 0.0
    if all_true_labels:
        cm = classification_metrics(all_true_labels, all_prediction_labels)
        acc = cm["accuracy"]
        precision = cm["precision"]
        rec = cm["recall"]
        f1 = cm["f1"]

    return {
        "mean_precision_at_k": mean_precision,
        "mean_ap": mean_ap,
        "map": mean_ap,
        "retrieval_map": retrieval_map,
        "accuracy": acc,
        "precision": precision,
        "recall": rec,
        "f1": f1
    }


def run_model_evaluations(query_file, top_k=5):
    # Load entire JSON
    with open(query_file, 'r', encoding='utf-8') as f:
        all_queries = json.load(f)

    # Split queries by model
    lstm_queries = [q for q in all_queries if q.get('model') == 'lstm']
    bert_queries = [q for q in all_queries if q.get('model') == 'bert']

    lstm_metrics = evaluate_queries(lstm_queries, top_k=top_k)
    bert_metrics = evaluate_queries(bert_queries, top_k=top_k)

    print("=== LSTM RESULTS ===")
    print(f" Accuracy: {lstm_metrics['accuracy']:.3f}")
    print(f" F1: {lstm_metrics['f1']:.3f}")
    print(f" Mean Precision@{top_k}: {lstm_metrics['mean_precision_at_k']:.3f}")
    print(f" MAP: {lstm_metrics['map']:.3f}")

    print("\n=== BERT RESULTS ===")
    print(f" Accuracy: {bert_metrics['accuracy']:.3f}")
    print(f" F1: {bert_metrics['f1']:.3f}")
    print(f" Mean Precision@{top_k}: {bert_metrics['mean_precision_at_k']:.3f}")
    print(f" MAP: {bert_metrics['map']:.3f}")

    combined_retrievals = {}
    # Merge entries from lstm retrievals
    for key, model_dict in lstm_metrics["retrieval_map"].items():
        combined_retrievals.setdefault(key, {}).update(model_dict)
    # Merge entries from bert retrievals
    for key, model_dict in bert_metrics["retrieval_map"].items():
        combined_retrievals.setdefault(key, {}).update(model_dict)

    overlap_ratios = []
    for key, model_dict in combined_retrievals.items():
        lstm_list = model_dict["lstm"]
        bert_list = model_dict["bert"]
        overlap_count = len(set(lstm_list).intersection(bert_list))
        overlap_ratio = overlap_count / top_k
        overlap_ratios.append(overlap_ratio)

    if overlap_ratios:
        mean_overlap = sum(overlap_ratios) / len(overlap_ratios)
        print(f"\n=== Overlap Results ===")
        print(f" Average Overlap Ratio (top-{top_k}): {mean_overlap:.3f}")
    else:
        print("\nNo matching queries found for overlap computation.")


if __name__ == "__main__":
    run_model_evaluations("evaluation-data/search_queries.json", top_k=5)
