from sentence_transformers import CrossEncoder
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
import json
import os


def find_models_to_evaluate(base_path):
    """
    Search for models that have ranking files and build a list of model paths.
    
    Args:
        base_path (str): The base directory containing model folders
        
    Returns:
        list: A list of model paths in the format "model_dir/checkpoint_name"
    """
    import os
    import glob
    import re
    
    model_paths = []
    models_with_rankings = set()
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Base path {base_path} does not exist")
        return []
    
    # List all model directories
    for model_dir in os.listdir(base_path):
        model_full_path = os.path.join(base_path, model_dir)
        
        if not os.path.isdir(model_full_path):
            continue
            
        # Check for rankings folder
        rankings_dir = os.path.join(model_full_path, "rankings")
        if os.path.exists(rankings_dir):
            # Find all .trec files in the rankings directory
            trec_files = glob.glob(os.path.join(rankings_dir, "*.trec"))
            
            for trec_file in trec_files:
                # Extract checkpoint name from trec filename 
                # Pattern: reranked.{checkpoint_name}.datetime.trec
                filename = os.path.basename(trec_file)
                match = re.search(r'reranked\.(.+?)\.[\d-]+', filename)
                
                if match:
                    checkpoint_name = match.group(1)
                    # Add to models with rankings set to avoid duplicates
                    models_with_rankings.add(f"{model_dir}/{checkpoint_name}")
    
    print(f"Found {len(models_with_rankings)} models to evaluate")
    for path in models_with_rankings:
        print(f"  - {path}")
    
    return list(models_with_rankings)
    

def calculate_mrr_at_10(model_path: str, parent_dir: str, num_queries_batch: int = 32) -> None:
    """
    Calculate MRR@10 with batched processing for better GPU utilization
    
    Args:
        model_path: Path to the model directory
        num_queries_batch: Number of queries to process in a single batch
    """
    base_model_path, ext = model_path.split("/")
    try:
        # check if the eval results already exist
        if os.path.exists(f"{parent_dir}/{base_model_path}/results/mrr_at_10_{ext}.json"):
            print(f"Results already exist for {model_path}, skipping evaluation.")
            return
    except Exception as e:
        print(f"Error checking for existing results: {e}")
        return

    # load the 10k-query development set
    print("Loading MS MARCO development dataset...")
    dev = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

    # instantiate your model
    model = CrossEncoder(parent_dir + model_path, device="cuda:0", trust_remote_code=True)

    all_reciprocal_ranks = []
    total_queries = len(dev)
    
    # Process queries in batches
    for batch_start in tqdm(range(0, total_queries, num_queries_batch), desc="Processing query batches"):
        batch_end = min(batch_start + num_queries_batch, total_queries)
        batch_items = dev[batch_start:batch_end]
        
        # Collect all query-passage pairs and track boundaries
        all_pairs = []
        query_passage_counts = []
        
        queries = batch_items['query']
        passages = batch_items['passages']
        for i in range(len(queries)):
            q = queries[i]
            texts = passages[i]['passage_text']
            
            # Create pairs for this query
            pairs = [(q, txt) for txt in texts]
            all_pairs.extend(pairs)
            
            # Track number of passages for this query
            query_passage_counts.append(len(pairs))
        
        # Score all pairs in the batch at once
        if all_pairs:
            all_scores = model.predict(all_pairs)
            
            # Process scores for each query
            score_start = 0
            for i, passage_count in enumerate(query_passage_counts):
                # Extract scores for current query
                query_index = batch_start + i
                item = dev[query_index]
                
                scores = all_scores[score_start:score_start + passage_count]
                labels = item["passages"]["is_selected"]
                
                # Rank passages by descending score
                ranked_idx = np.argsort(scores)[::-1]
                
                # Find first relevant in top 10
                rr = 0.0
                for rank, idx in enumerate(ranked_idx[:10], start=1):
                    if labels[idx] == 1:
                        rr = 1.0 / rank
                        break
                
                all_reciprocal_ranks.append(rr)
                score_start += passage_count
    
    # Compute MRR@10
    mrr_at_10 = np.mean(all_reciprocal_ranks)
    print(f"MRR@10 on MS MARCO dev: {mrr_at_10:.4f}")
    print(f"Processed {len(all_reciprocal_ranks)} queries with batch size {num_queries_batch}")

    os.makedirs(f"{parent_dir}/{base_model_path}/results", exist_ok=True)
    with open(f"{parent_dir}/{base_model_path}/results/mrr_at_10_{ext}.json", "w") as f:
        json.dump({"mrr_at_10": mrr_at_10}, f, indent=4)
    
    print(f"Results saved to {parent_dir}/{base_model_path}/results/mrr_at_10_{ext}.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate MRR@10 for a given model path")
    parser.add_argument("--parent_dir", type=str, help="Parent directory of the model path")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Number of queries to process in a single batch")
    args = parser.parse_args()

    all_models_to_evaluate = find_models_to_evaluate(args.parent_dir)


    for model_path in all_models_to_evaluate:
        start = time.time()
        calculate_mrr_at_10(model_path, args.parent_dir, num_queries_batch=args.batch_size)
        end = time.time()
        print(f"Time taken for {model_path} is {end - start}s")
