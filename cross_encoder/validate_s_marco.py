import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.metrics import ndcg_score, accuracy_score, precision_score, recall_score, f1_score

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def calculate_mrr_at_k(scores, labels, k=10):
    """Calculate Mean Reciprocal Rank at k"""
    # Get sorted indices in descending order of scores
    sorted_indices = np.argsort(-np.array(scores))[:k]  # Only consider top k
    
    # Find position of relevant documents
    for rank, idx in enumerate(sorted_indices, 1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0

def calculate_map(scores, labels):
    """Calculate Mean Average Precision"""
    sorted_indices = np.argsort(-np.array(scores))
    
    relevant_count = 0
    sum_precision = 0.0
    
    for rank, idx in enumerate(sorted_indices, 1):
        if labels[idx] == 1:
            relevant_count += 1
            sum_precision += relevant_count / rank
    
    # If there are no relevant documents, return 0
    if sum(labels) == 0:
        return 0.0
    
    return sum_precision / sum(labels)

def calculate_ndcg(scores, labels, k=10):
    """Calculate NDCG@k"""
    # NDCG requires at least 2 documents to compare
    if len(scores) < 2:
        # For a single document, if it's relevant (label=1) return 1.0, else 0.0
        return float(labels[0]) if len(labels) > 0 else 0.0
    
    # Normal case with multiple documents
    return ndcg_score([labels], [scores], k=min(k, len(scores)))

def calculate_binary_metrics(predictions, labels, threshold=0.8):
    """Calculate binary classification metrics using a threshold"""
    # Convert predictions to scalar values if they are arrays
    binary_preds = []
    for score in predictions:
        # Handle case where score is a numpy array
        if isinstance(score, np.ndarray):
            # Take the first element if it's an array
            score_value = float(score[0]) if score.size > 0 else 0.0
        else:
            score_value = float(score)
        
        binary_preds.append(1 if score_value >= threshold else 0)
    
    # Convert labels to binary using threshold
    binary_labels = [1 if label >= 0.8 else 0 for label in labels]
    
    accuracy = accuracy_score(binary_labels, binary_preds)
    precision = precision_score(binary_labels, binary_preds, zero_division=0)
    recall = recall_score(binary_labels, binary_preds, zero_division=0)
    f1 = f1_score(binary_labels, binary_preds, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def validate_on_saved_dataset(model_path, eval_dataset_path, batch_size=32, threshold=0.8):
    """
    Validate a cross-encoder model on a saved validation dataset.
    
    Args:
        model_path: Path to the trained model
        eval_dataset_path: Path to the saved validation dataset
        batch_size: Batch size for prediction
        threshold: Threshold for binary classification (0.0-1.0)
    
    Returns:
        Dictionary with validation metrics
    """
    logging.info(f"Loading model from {model_path}")
    model = CrossEncoder(model_path)
    
    logging.info(f"Loading validation dataset from {eval_dataset_path}")
    try:
        eval_dataset = load_from_disk(eval_dataset_path)
        logging.info(f"Dataset loaded successfully with {len(eval_dataset)} examples")
        logging.info(f"Dataset features: {eval_dataset.features}")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise
    
    # Track examples by query
    query_groups = {}
    
    # Group by query to analyze structure
    logging.info("Grouping examples by query")
    for idx in range(len(eval_dataset)):
        example = eval_dataset[idx]
        query = example['query']
        passage = example['passage']
        
        # Handle different score formats
        if 'score' in example:
            score = float(example['score']) if not isinstance(example['score'], (list, np.ndarray)) else float(example['score'][0])
        elif 'label' in example:
            score = float(example['label']) if not isinstance(example['label'], (list, np.ndarray)) else float(example['label'][0])
        else:
            score = 0.0  # Default if no score field found
        
        if query not in query_groups:
            query_groups[query] = {
                'passages': [], 
                'labels': []
            }
        
        query_groups[query]['passages'].append(passage)
        query_groups[query]['labels'].append(score)
    
    # Analyze dataset structure
    passage_counts = {}
    relevance_distribution = {0: 0, 1: 0}
    
    for query, group in query_groups.items():
        num_passages = len(group['passages'])
        if num_passages not in passage_counts:
            passage_counts[num_passages] = 0
        passage_counts[num_passages] += 1
        
        for label in group['labels']:
            if label >= 0.8:
                relevance_distribution[1] += 1
            else:
                relevance_distribution[0] += 1
    
    logging.info(f"Found {len(query_groups)} unique queries")
    logging.info(f"Passage counts per query: {passage_counts}")
    logging.info(f"Relevance distribution: {relevance_distribution}")
    
    # For single-passage datasets, use binary classification metrics
    if passage_counts.get(1, 0) / len(query_groups) > 0.9:  # If over 90% are single-passage
        logging.info("Dataset consists primarily of single-passage queries, using binary classification evaluation")
        
        all_labels = []
        all_predictions = []
        
        # Process all examples as binary classification
        for query, group in tqdm(query_groups.items()):
            passages = group['passages']
            labels = group['labels']
            
            # Create query-passage pairs
            pairs = [(query, passage) for passage in passages]
            
            # Get predictions
            try:
                scores = model.predict(pairs)
                
                # Handle different return types from model.predict
                if isinstance(scores, np.ndarray):
                    # If it's a 2D array, flatten it
                    if len(scores.shape) > 1:
                        scores = scores.flatten()
                    # If it's a single value, convert to list
                    elif len(scores.shape) == 0:
                        scores = [float(scores)]
                    else:
                        scores = scores.tolist()
                elif not isinstance(scores, list):
                    scores = [scores]
                
                all_labels.extend(labels)
                all_predictions.extend(scores)
                
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                continue
        
        # Calculate binary metrics
        binary_metrics = calculate_binary_metrics(all_predictions, all_labels, threshold=threshold)
        
        # Log results
        logging.info(f"Binary Classification Results on {len(all_labels)} examples:")
        logging.info(f"Accuracy: {binary_metrics['accuracy']:.4f}")
        logging.info(f"Precision: {binary_metrics['precision']:.4f}")
        logging.info(f"Recall: {binary_metrics['recall']:.4f}")
        logging.info(f"F1 Score: {binary_metrics['f1']:.4f}")
        
        return {
            "accuracy": binary_metrics['accuracy'],
            "precision": binary_metrics['precision'],
            "recall": binary_metrics['recall'],
            "f1": binary_metrics['f1'],
            "num_examples": len(all_labels),
            "evaluation_type": "binary_classification"
        }
    
    # For ranking datasets (multiple passages per query)
    else:
        logging.info("Dataset has multiple passages per query, using ranking metrics")
        
        # Metrics to track
        mrr_values = []
        map_values = []
        ndcg_values = []
        
        # Process each query group
        skipped = 0
        processed = 0
        
        for query, group in tqdm(query_groups.items()):
            passages = group['passages']
            labels = group['labels']
            
            # Skip groups with no positive examples or only one document
            # Convert scores to binary using threshold
            binary_labels = [1 if score >= threshold else 0 for score in labels]
            
            if 1 not in binary_labels or len(passages) < 2:
                skipped += 1
                continue
            
            # Create query-passage pairs
            pairs = [(query, passage) for passage in passages]
            
            # Get predictions in batches for efficiency
            scores = []
            try:
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i:i+batch_size]
                    batch_scores = model.predict(batch_pairs)
                    
                    # Handle different return types
                    if isinstance(batch_scores, np.ndarray):
                        # If we got a 2D array, flatten it
                        if len(batch_scores.shape) > 1:
                            batch_scores = batch_scores.flatten()
                        # If single value, convert to list
                        elif len(batch_scores.shape) == 0:
                            batch_scores = [float(batch_scores)]
                        else:
                            batch_scores = batch_scores.tolist()
                    elif not isinstance(batch_scores, list):
                        batch_scores = [batch_scores]
                    
                    # Extract scalar values
                    batch_scores_scalar = []
                    for score in batch_scores:
                        if isinstance(score, np.ndarray):
                            batch_scores_scalar.append(float(score[0]) if score.size > 0 else 0.0)
                        else:
                            batch_scores_scalar.append(float(score))
                    
                    scores.extend(batch_scores_scalar)
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                skipped += 1
                continue
            
            # Calculate metrics using binary labels
            mrr = calculate_mrr_at_k(scores, binary_labels, k=10)
            map_score = calculate_map(scores, binary_labels)
            ndcg = calculate_ndcg(scores, binary_labels, k=10)
            
            mrr_values.append(mrr)
            map_values.append(map_score)
            ndcg_values.append(ndcg)
            processed += 1
            
            # Log progress occasionally
            if processed % 100 == 0:
                logging.info(f"Processed {processed} query groups, skipped {skipped}")
        
        # Calculate average metrics
        avg_mrr = np.mean(mrr_values) if mrr_values else 0.0
        avg_map = np.mean(map_values) if map_values else 0.0
        avg_ndcg = np.mean(ndcg_values) if ndcg_values else 0.0
        
        # Print results
        logging.info(f"Ranking Results on {len(mrr_values)} queries:")
        logging.info(f"MRR@10: {avg_mrr:.4f}")
        logging.info(f"MAP: {avg_map:.4f}")
        logging.info(f"NDCG@10: {avg_ndcg:.4f}")
        logging.info(f"Skipped {skipped} queries (no relevant passages or single-doc queries)")
        
        return {
            "mrr@10": avg_mrr,
            "map": avg_map,
            "ndcg@10": avg_ndcg,
            "num_queries": len(mrr_values),
            "skipped_queries": skipped,
            "evaluation_type": "ranking"
        }

def evaluate_model(model_path, eval_dataset_path, output_path=None, batch_size=32, threshold=0.8):
    """
    Evaluate a model and optionally save the results to a file.
    
    Args:
        model_path: Path to the trained model
        eval_dataset_path: Path to the saved validation dataset
        output_path: Path to save results (optional)
        batch_size: Batch size for prediction
        threshold: Threshold for binary classification
    """
    # Run validation
    metrics = validate_on_saved_dataset(model_path, eval_dataset_path, batch_size, threshold)
    
    # Print summary
    print("\nValidation Summary:")
    if metrics.get("evaluation_type") == "binary_classification":
        print(f"Evaluation Type: Binary Classification")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Number of examples evaluated: {metrics['num_examples']}")
    else:
        print(f"Evaluation Type: Ranking")
        print(f"MRR@10: {metrics['mrr@10']:.4f}")
        print(f"MAP: {metrics['map']:.4f}")
        print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
        print(f"Number of queries evaluated: {metrics['num_queries']}")
        print(f"Queries skipped: {metrics['skipped_queries']}")
    
    # Save results if output path is provided
    if output_path:
        import json
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a cross-encoder model on a separate validation dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--output_path", type=str, help="Path to save results (optional)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for predictions")
    parser.add_argument("--threshold", type=float, default=0.8, help="Threshold for binary classification")
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.eval_dataset, args.output_path, args.batch_size, args.threshold)