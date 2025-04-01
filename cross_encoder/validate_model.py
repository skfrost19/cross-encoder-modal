import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.metrics import ndcg_score

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def calculate_mrr(scores, labels):
    """Calculate Mean Reciprocal Rank"""
    # Get sorted indices in descending order of scores
    sorted_indices = np.argsort(-np.array(scores))
    
    # Find position of relevant documents
    for rank, idx in enumerate(sorted_indices, 1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0

def calculate_ndcg(scores, labels, k=10):
    """Calculate NDCG@k"""
    return ndcg_score([labels], [scores], k=min(k, len(scores)))

def group_by_query(dataset):
    """Group passages by query ID for proper ranking evaluation"""
    query_groups = {}
    
    for i in range(len(dataset['query'])):
        query = dataset['query'][i]
        if query not in query_groups:
            query_groups[query] = {
                'passages': [],
                'labels': [],
                'query': query
            }
        
        query_groups[query]['passages'].append(dataset['passage'][i])
        query_groups[query]['labels'].append(dataset['score'][i])
    
    return list(query_groups.values())

def validate_model(model_path, eval_dataset_path, batch_size=32):
    """
    Validate a cross-encoder model on a validation set.
    
    Args:
        model_path: Path to the trained model
        eval_dataset_path: Path to the validation dataset
        batch_size: Batch size for prediction
    
    Returns:
        Dictionary with validation metrics
    """
    logging.info(f"Loading model from {model_path}")
    model = CrossEncoder(model_path)
    
    logging.info(f"Loading validation dataset from {eval_dataset_path}")
    eval_dataset = load_from_disk(eval_dataset_path)
    
    # Group by query for proper ranking evaluation
    logging.info("Grouping validation data by query")
    query_groups = group_by_query(eval_dataset)
    
    # Metrics to track
    mrr_values = []
    ndcg_at_10_values = []
    ndcg_at_3_values = []
    ndcg_at_1_values = []
    
    # Evaluate each query group
    logging.info(f"Evaluating on {len(query_groups)} queries")
    for group in tqdm(query_groups):
        query = group['query']
        passages = group['passages']
        labels = group['labels']
        
        # Skip groups with no positive examples
        if 1.0 not in labels:
            continue
        
        # Create query-passage pairs
        pairs = [(query, passage) for passage in passages]
        
        # Get predictions
        scores = model.predict(pairs)
        
        # Calculate metrics
        mrr = calculate_mrr(scores, labels)
        ndcg10 = calculate_ndcg(scores, labels, k=10)
        ndcg3 = calculate_ndcg(scores, labels, k=3)
        ndcg1 = calculate_ndcg(scores, labels, k=1)
        
        mrr_values.append(mrr)
        ndcg_at_10_values.append(ndcg10)
        ndcg_at_3_values.append(ndcg3)
        ndcg_at_1_values.append(ndcg1)
    
    # Calculate average metrics
    avg_mrr = np.mean(mrr_values)
    avg_ndcg10 = np.mean(ndcg_at_10_values)
    avg_ndcg3 = np.mean(ndcg_at_3_values)
    avg_ndcg1 = np.mean(ndcg_at_1_values)
    
    # Print results
    logging.info(f"Validation Results on {len(mrr_values)} queries with relevant passages:")
    logging.info(f"MRR: {avg_mrr:.4f}")
    logging.info(f"NDCG@10: {avg_ndcg10:.4f}")
    logging.info(f"NDCG@3: {avg_ndcg3:.4f}")
    logging.info(f"NDCG@1: {avg_ndcg1:.4f}")
    
    return {
        "mrr": avg_mrr,
        "ndcg@10": avg_ndcg10,
        "ndcg@3": avg_ndcg3,
        "ndcg@1": avg_ndcg1,
        "num_queries": len(mrr_values)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a cross-encoder model on MS MARCO validation set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for predictions")
    
    args = parser.parse_args()
    
    # Run validation
    metrics = validate_model(args.model_path, args.eval_dataset, args.batch_size)
    
    # Print summary
    print("\nValidation Summary:")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"NDCG@3: {metrics['ndcg@3']:.4f}")
    print(f"NDCG@1: {metrics['ndcg@1']:.4f}")
    print(f"Number of queries evaluated: {metrics['num_queries']}")