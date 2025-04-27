import ir_datasets
import pandas as pd
import numpy as np
import json
import pickle
import argparse
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime
from sentence_transformers import CrossEncoder
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import time

# Define all available trec_eval metrics with correct naming
TREC_EVAL_METRICS = [
    "map",
    "P_5", "P_10", "P_20",
    "recall_5", "recall_10", "recall_20", "recall_100", "recall_1000",
    "ndcg", "ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20", "ndcg_cut_100",
    "recip_rank",
    "Rprec",
    "bpref",
    "iprec_at_recall_0.0", "iprec_at_recall_0.5", "iprec_at_recall_1.0",
    "gm_map",
    "set_P", "set_recall", "set_F", "set_relative_P", "set_map",
    "success_1", "success_5", "success_10",
    "map_avgjg", "P_avgjg"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate cross-encoder models on TREC DL 2019")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="skfrost19/reranker-gte-multilingual-base-msmarco-bce-ep-2",
                        help="Cross-encoder model name or path")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run model on (cuda:0, cpu, etc.)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for inference")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="msmarco-passage/trec-dl-2019",
                        help="IR dataset to evaluate (default: TREC DL 2019)")
    parser.add_argument("--max_passages", type=int, default=1000,
                        help="Maximum number of passages to retrieve per query")
    
    # File paths (only results saving is kept)
    parser.add_argument("--runs_dir", type=str, default="runs",
                        help="Directory to (temporarily) store run files")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to store evaluation results")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Directory to store cached passages and other data")
    parser.add_argument("--bm25_run_path", type=str, default=None,
                        help="Path to existing BM25 run file (not used since saving is removed)")
    parser.add_argument("--force_bm25", action="store_true",
                        help="Force running BM25 even if results file exists")
    parser.add_argument("--rerank_only", action="store_true",
                        help="Only perform reranking using an existing BM25 run file")
    
    # Evaluation parameters
    parser.add_argument("--use_all_metrics", action="store_true",
                        help="Use all available TREC metrics for evaluation")
    parser.add_argument("--metrics", type=str, nargs="+", 
                        default=["ndcg", "ndcg_cut.5", "ndcg_cut.10", "ndcg_cut.20", "ndcg_cut.100", "map", 
                                 "recall.5", "recall.10", "recall.20", "recall.100", "recall.1000", 
                                 "recip_rank", "P.10"],
                        help="TREC eval metrics to report (ignored if --use_all_metrics is set)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.rerank_only and args.bm25_run_path is None:
        parser.error("--rerank_only requires --bm25_run_path")
    
    return args

def run_trec_eval(qrels_path, run_path, metrics):
    """Run trec_eval and parse the results"""

    cmd = ["./trec_eval/trec_eval", "-m","all_trec", qrels_path, run_path]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Parse results
    lines = result.stdout.strip().split('\n')
    results = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) != 3:
            continue
            
        metric, qid, value = parts
        
        # Only keep the 'all' summary values
        if qid != 'all':
            continue
            
        try:
            value = float(value)
            results[metric] = value
        except ValueError:
            # Keep non-numeric values as strings
            results[metric] = value
    
    # run again to get the detailed results
    cmd = ["./trec_eval/trec_eval", "-m","all_trec", "-q", qrels_path, run_path]
    
    result_detailed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("Result of trec_eval:")
    print("*************************************************************")
    print(result_detailed.stdout)
    print("*************************************************************")
    print("Error of trec_eval:")
    print("*************************************************************")
    print(result_detailed.stderr)
    print("*************************************************************")    
    # Parse results
    lines = result_detailed.stdout.strip().split('\n')
    detailed_results = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) != 3:
            continue
            
        metric, qid, value = parts
        
        # Only keep the 'all' summary values
        if qid == 'all':
            continue
            
        try:
            value = float(value)
            detailed_results[metric] = value
        except ValueError:
            # Keep non-numeric values as strings
            detailed_results[metric] = value

    return results, detailed_results

def main():
    args = parse_args()
    
    if args.use_all_metrics:
        metrics_to_use = TREC_EVAL_METRICS
    else:
        metrics_to_use = args.metrics
    
    # Create only the directories needed (results, temporary run files, and cache for in-memory use)
    os.makedirs(args.runs_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_short_name = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    
    # Always run BM25 retrieval (ignoring any pre-saved file)
    results_path = os.path.join(args.results_dir, f"eval.{model_short_name}.{timestamp}.json")
    detailed_results_path = os.path.join(args.results_dir, f"eval.{model_short_name}.{timestamp}.detailed.json")

    model_path_local = args.results_dir.replace("/results", "")
    
    # Ensure the directory exists
    reranked_dir = os.path.join(model_path_local, "rankings")
    os.makedirs(reranked_dir, exist_ok=True)

    reranked_run_path = os.path.join(model_path_local + "/rankings", f"reranked.{model_short_name}.{timestamp}.trec")
    
    print(f"Loading dataset {args.dataset}...")
    dataset = ir_datasets.load(args.dataset)
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    print(f"Loaded {len(queries)} queries")
    
    # Run BM25 retrieval
    print("Loading BM25 index...")
    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    if searcher is None:
        print("Error loading prebuilt index 'msmarco-v1-passage'")
        exit(1)
    
    print("Running BM25 retrieval...")
    bm25_results = []
    for qid in tqdm(queries, desc="BM25 retrieval"):
        query_text = queries[qid]
        hits = searcher.search(query_text, k=args.max_passages)
        for rank, hit in enumerate(hits):
            bm25_results.append({
                'qid': qid,
                'Q0': 'Q0',
                'doc_id': hit.docid,
                'rank': rank + 1,
                'score': hit.score,
                'run': 'bm25'
            })
    bm25_df = pd.DataFrame(bm25_results)
    print("BM25 retrieval complete.")
    
    print("Loading msmarco-passage docs_store...")
    docs = ir_datasets.load("msmarco-passage").docs_store()
    
    try:
        if len(os.listdir(reranked_dir)) > 0:
            print(f"Reranked run file(s) already exists, loading...")
            reranked_run_path = sorted(Path(reranked_dir).glob(f"reranked.{model_short_name}.*.trec"))[-1]
        else:
            raise FileNotFoundError
    except:
        print(f"Reranked run file(s) do not exist, proceeding with reranking.")
        # Rerank BM25 results using the cross-encoder
        print("Reranking with cross-encoder...")
        # Initialize cross-encoder model
        print(f"Loading model {args.model_name}...")
        model = CrossEncoder(args.model_name, trust_remote_code=True, device=args.device)
        print(f"Reranked run file {reranked_run_path} does not exist. Proceeding with reranking.")
        with open(reranked_run_path, "w") as f_out:
            for qid in tqdm(queries, desc="Reranking queries"):
                query_text = queries[qid]
                query_docs_df = bm25_df[bm25_df["qid"] == qid]
                query_docs = query_docs_df.sort_values("rank")["doc_id"].tolist()[:args.max_passages]
        
                passages = []
                valid_doc_ids = []
                for doc_id in query_docs:
                    try:
                        doc = docs.get(doc_id)
                        if doc is not None:
                            passages.append(doc.text)
                            valid_doc_ids.append(doc_id)
                    except Exception as e:
                        continue
        
                if not passages:
                    print(f"Warning: No passages found for query {qid}")
                    continue
        
                pairs = [(query_text, passage) for passage in passages]
                print(f"Reranking {len(pairs)} pairs for query {qid}")
                scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
        
                sorted_docs = sorted(zip(valid_doc_ids, scores), key=lambda x: x[1], reverse=True)
        
                for rank, (doc_id, score) in enumerate(sorted_docs, 1):
                    line = f"{qid} Q0 {doc_id} {rank} {score} {model_short_name}\n"
                    f_out.write(line)
        f_out.close()
        print(f"Reranked run file saved to {reranked_run_path}")
    # Install trec_eval if not already present
    if not os.path.exists("trec_eval"):
        print("Installing trec_eval...")
        subprocess.run(["git", "clone", "https://github.com/usnistgov/trec_eval.git"], check=True)
        subprocess.run("cd trec_eval && make", shell=True, check=True)
    
    print("\nRunning evaluation...")
    qrels_path = dataset.qrels_path()
    
    try:
        avg_results, detailed_res = run_trec_eval(qrels_path, reranked_run_path, metrics_to_use)
        
        # save both the results
        if avg_results:
            with open(results_path, "w") as f:
                json.dump(avg_results, f, indent=2)
        
        if detailed_res:
            with open(detailed_results_path, "w") as f:
                json.dump(detailed_res, f, indent=2)
        sys.exit(0)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running trec_eval: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)

if __name__ == "__main__":
    main()