import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.metrics import ndcg_score

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def infer_model(model_path, query, passages):
    """
    Rerank passages using a cross-encoder model.
    
    Args:
        model_path: Path to the trained model
        query: The search query
        passages: List of passage texts to rerank
    
    Returns:
        List of tuples (passage, score) sorted by score in descending order
    """
    logging.info(f"Loading model from {model_path}")
    model = CrossEncoder(model_path)
    
    # Create query-passage pairs
    pairs = [(query, passage) for passage in passages]
    
    # Get predictions
    scores = model.predict(pairs)
    
    # Sort passages by score
    sorted_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    
    return sorted_passages

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

def validate_model(model_path, batch_size=32, num_samples=2000):
    """
    Validate a cross-encoder model on MS MARCO validation set.
    
    Args:
        model_path: Path to the trained model
        batch_size: Batch size for prediction
        num_samples: Number of validation samples to use
    
    Returns:
        Dictionary with validation metrics
    """
    logging.info(f"Loading model from {model_path}")
    model = CrossEncoder(model_path)
    
    logging.info("Loading MS MARCO validation data from Hugging Face")
    
    # Load dev set from MS MARCO
    dev_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
    
    # Create a passage dictionary for lookup
    passage_dict = {}
    for item in dev_dataset:
        if 'passages' in item and 'passage_text' in item['passages']:
            for i, passage_text in enumerate(item['passages']['passage_text']):
                # Create a unique passage ID by combining query_id and position
                passage_id = f"{item['query_id']}_{i}"
                passage_dict[passage_id] = passage_text

    # Limit number of queries if needed
    if num_samples and num_samples < len(dev_dataset):
        dev_dataset = dev_dataset.select(range(num_samples))
    
    logging.info(f"Evaluating on {len(dev_dataset)} queries")
    
    # Metrics to track
    mrr_values = []
    map_values = []
    ndcg_values = []
    
    # Count skipped queries
    skipped_queries = 0
    processed_queries = 0
    
    # Process each query
    for item in tqdm(dev_dataset):
        # Skip if no passages or no is_selected info
        if not item.get('passages') or 'is_selected' not in item['passages']:
            skipped_queries += 1
            continue
            
        query = item['query']
        is_selected = item['passages']['is_selected']
        passage_texts = item['passages']['passage_text']
        
        # Skip if no relevant passages found
        if 1 not in is_selected or len(passage_texts) < 2:
            skipped_queries += 1
            continue
        
        # Convert to list format for processing
        labels = [float(x) for x in is_selected]
        passages = passage_texts
        
        # Create query-passage pairs
        pairs = [(query, passage) for passage in passages]
        
        # Get predictions in batches for efficiency
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch_scores = model.predict(batch_pairs)
            if isinstance(batch_scores, np.ndarray) and len(batch_scores.shape) == 0:
                # Handle case where batch_scores is a single number
                batch_scores = [float(batch_scores)]
            elif not isinstance(batch_scores, list):
                batch_scores = batch_scores.tolist()
            scores.extend(batch_scores)
        
        # Calculate metrics
        mrr = calculate_mrr_at_k(scores, labels, k=10)
        map_score = calculate_map(scores, labels)
        ndcg = calculate_ndcg(scores, labels, k=10)
        
        mrr_values.append(mrr)
        map_values.append(map_score)
        ndcg_values.append(ndcg)
        processed_queries += 1
        
        # Log progress occasionally
        if processed_queries % 100 == 0:
            logging.info(f"Processed {processed_queries} queries so far")
    
    # Calculate average metrics
    avg_mrr = np.mean(mrr_values) if mrr_values else 0.0
    avg_map = np.mean(map_values) if map_values else 0.0
    avg_ndcg = np.mean(ndcg_values) if ndcg_values else 0.0
    
    # Print results
    logging.info(f"Validation Results on {len(mrr_values)} queries:")
    logging.info(f"MRR@10: {avg_mrr:.4f}")
    logging.info(f"MAP: {avg_map:.4f}")
    logging.info(f"NDCG@10: {avg_ndcg:.4f}")
    logging.info(f"Skipped {skipped_queries} queries without relevant passages")
    
    return {
        "mrr@10": avg_mrr,
        "map": avg_map,
        "ndcg@10": avg_ndcg,
        "num_queries": len(mrr_values),
        "skipped_queries": skipped_queries
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a cross-encoder model on MS MARCO validation set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for predictions")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of validation queries to use")
    
    args = parser.parse_args()
    
    # Run validation
    metrics = validate_model(args.model_path, args.batch_size, args.num_samples)
    
    # Print summary
    print("\nValidation Summary:")
    print(f"MRR@10: {metrics['mrr@10']:.4f}")
    print(f"MAP: {metrics['map']:.4f}")
    print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"Number of queries evaluated: {metrics['num_queries']}")
    print(f"Queries skipped: {metrics['skipped_queries']}")

    # Test inference on sample query
    query = "How many people live in Berlin?"
    passages = [
        "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        "Berlin is well known for its museums.",
        "In 2014, the city state Berlin had 37,368 live births (+6.6%), a record number since 1991.",
        "The urban area of Berlin comprised about 4.1 million people in 2014, making it the seventh most populous urban area in the European Union.",
        "The city of Paris had a population of 2,165,423 people within its administrative city limits as of January 1, 2019",
        "An estimated 300,000-420,000 Muslims reside in Berlin, making up about 8-11 percent of the population.",
        "Berlin is subdivided into 12 boroughs or districts (Bezirke).",
        "In 2015, the total labour force in Berlin was 1.85 million.",
        "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
        "Berlin has a yearly total of about 135 million day visitors, which puts it in third place among the most-visited city destinations in the European Union.",
    ]
    print("\nInference Test:")
    sorted_passages = infer_model(args.model_path, query, passages)
    for passage, score in sorted_passages:
        print(f"Score: {score:.4f} - {passage[:100]}...")