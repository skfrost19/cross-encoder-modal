import os
import json
import csv
import re
import glob
from pathlib import Path
from collections import OrderedDict

def extract_info_from_path(file_path):
    """Extract model, optimizer, and epoch info from file path"""
    # Convert to Path object for easier manipulation
    path = Path(file_path)
    
    # Extract components from path
    try:
        model = path.parts[-3]  # e.g., 'gte', 'mini_lm', 'modern_bert'
        optimizer = path.parts[-2]  # e.g., 'AdamW', 'Lion'
        
        # Extract epoch number from filename (ep1.json, ep2.json, etc.)
        epoch_match = re.search(r'ep(\d+)', path.stem)
        if epoch_match:
            epoch = epoch_match.group(1)
        else:
            # Fallback to filename without extension
            epoch = path.stem
        
        return model, optimizer, epoch
    except IndexError:
        print(f"Warning: Could not extract info from path: {file_path}")
        return "unknown", "unknown", "unknown"

def merge_metrics_to_csv(base_dir, output_file):
    """
    Scan all JSON files in the directory structure and merge them into a single CSV,
    including ALL fields from the JSON files.
    
    Args:
        base_dir: Base directory containing evaluation results
        output_file: Path to output CSV file
    """
    # Find all JSON files recursively
    json_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    print(f"Found {len(json_files)} JSON files to process")
    
    # Create a set of all unique field names across all JSON files
    all_fields = set(["model", "optimizer", "epoch"])  # Start with our metadata fields
    
    # First pass: collect all possible field names
    print("Collecting all metric fields...")
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Add all keys from this JSON to our set of field names
                all_fields.update(data.keys())
        except Exception as e:
            print(f"Error reading fields from {file_path}: {e}")
    
    # Convert to ordered list and ensure metadata fields come first
    fields = ["model", "optimizer", "epoch"] + sorted(list(all_fields - {"model", "optimizer", "epoch"}))
    print(f"Found {len(fields)} unique fields across all JSON files")
    
    # Prepare data for CSV
    results = []
    
    # Second pass: extract data for each file
    for file_path in json_files:
        try:
            # Skip the mrrs.json file at the root level, as it has a different structure
            if os.path.basename(file_path) == "mrrs.json" and len(Path(file_path).parts) == len(Path(base_dir).parts) + 1:
                print(f"Skipping root-level file: {file_path}")
                continue
                
            # Extract info from path
            model, optimizer, epoch = extract_info_from_path(file_path)
            
            # Read JSON data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Create a row with path info and all metrics
            row = OrderedDict()
            row["model"] = model
            row["optimizer"] = optimizer
            row["epoch"] = epoch
            
            # Add all metrics from the JSON file
            for field in fields:
                if field in ["model", "optimizer", "epoch"]:
                    continue  # Skip fields we already added
                
                # Get the value directly from the JSON
                value = data.get(field)
                row[field] = value
            
            results.append(row)
            print(f"Processed: {file_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Write to CSV
    if results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        print(f"Successfully wrote {len(results)} entries to {output_file}")
    else:
        print("No data to write to CSV.")

def main():
    # Set paths
    base_dir = "d:\\Thesis\\4th Sem\\cross_encoder\\eval_results"
    output_file = "d:\\Thesis\\4th Sem\\cross_encoder\\all_metrics.csv"
    
    # Run the merger
    merge_metrics_to_csv(base_dir, output_file)
    
    # Print metrics definitions for reference
    print("\nCommon Metrics Definitions:")
    definitions = {
        "num_q": "Number of queries evaluated",
        "num_ret": "Total number of documents retrieved across all queries",
        "num_rel": "Total number of relevant documents in the collection",
        "num_rel_ret": "Number of relevant documents retrieved",
        "map": "Mean Average Precision - overall effectiveness of ranking",
        "gm_map": "Geometric Mean MAP - less affected by outlier queries",
        "Rprec": "R-Precision - precision after R documents (R = number of relevant docs)",
        "bpref": "Binary Preference - measures ranking when judgments are incomplete",
        "recip_rank": "Reciprocal Rank - 1/rank of the first relevant document",
        "P_5": "Precision at 5 - fraction of top 5 documents that are relevant",
        "P_10": "Precision at 10 - fraction of top 10 documents that are relevant",
        "ndcg_cut_10": "NDCG@10 - normalized discounted cumulative gain at rank 10",
        "recall_10": "Recall at 10 - fraction of relevant documents found in top 10",
        "iprec_at_recall_0.00": "Interpolated precision at 0% recall level",
        "utility": "Linear utility measure of user satisfaction",
        "runid": "Original run identifier from evaluation",
        "success_1": "Whether at least one relevant document was found in top result",
        "success_5": "Whether at least one relevant document was found in top 5 results",
        "success_10": "Whether at least one relevant document was found in top 10 results"
    }
    
    for metric, definition in definitions.items():
        print(f"{metric}: {definition}")

if __name__ == "__main__":
    main()