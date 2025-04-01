#!/usr/bin/env python3
import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from tqdm import tqdm
import os
import time
import json
from sklearn.metrics import ndcg_score
from lion_pytorch import Lion  # Requires: pip install lion-pytorch

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a cross-encoder model for document ranking')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
parser.add_argument('--model_name', type=str, default="allenai/longformer-base-4096", help='Pretrained model name')
parser.add_argument('--output_dir', type=str, default="./longformer_crossencoder", help='Output directory for saving the model')
args = parser.parse_args()

# Configuration
MODEL_NAME = args.model_name
MAX_LENGTH = args.max_length
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.lr
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create timestamp for log files
timestamp = time.strftime("%Y%m%d_%H%M%S")
loss_log_file = f"{timestamp}_training_log.json"

# Print configuration
print(f"Using configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Max Length: {MAX_LENGTH}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Device: {DEVICE}")
print(f"  Log file: {loss_log_file}")

# Custom metric functions
def calculate_mrr(results):
    """Calculate Mean Reciprocal Rank"""
    reciprocal_ranks = []
    for query_id, (scores, labels) in results.items():
        indices = np.argsort(-np.array(scores))
        for rank, idx in enumerate(indices, 1):
            if labels[idx] == 1:
                reciprocal_ranks.append(1/rank)
                break
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0

def calculate_ndcg(results, k=10):
    """Calculate Normalized Discounted Cumulative Gain"""
    ndcgs = []
    for query_id, (scores, labels) in results.items():
        true_relevance = np.array(labels)
        pred_relevance = np.array(scores)
        ndcgs.append(ndcg_score([true_relevance], [pred_relevance], k=k))
    return np.mean(ndcgs)

# Load dataset and verify structure
dataset = load_dataset("ms_marco", "v1.1")
print("Dataset features:", dataset["train"].features)  # Verify actual columns

# Process more data, with a positive-to-negative ratio of 1:3
def balance_dataset(dataset, train_size=50000):
    # Get all examples with positive labels
    positive_examples = [i for i, example in enumerate(dataset) if example["label"] == 1]
    # Get all examples with negative labels
    negative_examples = [i for i, example in enumerate(dataset) if example["label"] == 0]
    
    # Calculate how many positive examples to use
    pos_count = min(len(positive_examples), train_size // 4)
    # Calculate how many negative examples to use (3x positives)
    neg_count = min(len(negative_examples), pos_count * 3)
    
    # Select random samples
    import random
    selected_positives = random.sample(positive_examples, pos_count)
    selected_negatives = random.sample(negative_examples, neg_count)
    
    # Combine and shuffle
    selected_indices = selected_positives + selected_negatives
    random.shuffle(selected_indices)
    
    return dataset.select(selected_indices[:train_size])

def flatten_dataset(examples):
    new_examples = {"query": [], "passage": [], "label": [], "query_id": []}
    for i in range(len(examples["query"])):
        # Check if passages exists and has the expected structure
        if "passages" in examples and "passage_text" in examples["passages"][i] and "is_selected" in examples["passages"][i]:
            passage_texts = examples["passages"][i]["passage_text"]
            is_selected_values = examples["passages"][i]["is_selected"]
            
            # Get query ID or use index as fallback
            query_id = examples.get("query_id", [i])[i]
            
            for j in range(len(passage_texts)):
                new_examples["query"].append(examples["query"][i])
                new_examples["passage"].append(passage_texts[j])
                new_examples["label"].append(is_selected_values[j])
                new_examples["query_id"].append(query_id)
    
    return new_examples

# Process dataset (using smaller subset for demonstration)
train_dataset = dataset["train"].select(range(1000)).map(
    flatten_dataset,
    batched=True,
    remove_columns=dataset["train"].column_names,
    batch_size=100
)
val_dataset = dataset["validation"].select(range(200)).map(
    flatten_dataset,
    batched=True,
    remove_columns=dataset["validation"].column_names,
    batch_size=100
)

# Tokenization function
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["query"],
        examples["passage"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

# Process datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

columns = ["input_ids", "attention_mask", "label", "query_id"]
train_dataset.set_format(type="torch", columns=columns)
val_dataset.set_format(type="torch", columns=columns)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model with long context support
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    attention_window=512  # For longformer-style attention
).to(DEVICE)

# Lion optimizer setup
optimizer = Lion(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

# Better scheduler
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * args.warmup_ratio)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Add weights to your BCE loss to emphasize positive examples
pos_weight = torch.tensor([3.0]).to(DEVICE)  # Weight positive examples 3x
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Initialize logs
training_log = {
    "config": {
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
    },
    "training": {
        "losses": [],
        "avg_loss_per_epoch": []
    },
    "validation": {
        "mrr": [],
        "ndcg": []
    }
}

# Training loop with metrics
best_mrr = 0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    
    # Training
    model.train()
    progress_bar = tqdm(train_loader, desc="Training")
    epoch_losses = []
    
    for batch in progress_bar:
        inputs = {
            "input_ids": batch["input_ids"].to(DEVICE),
            "attention_mask": batch["attention_mask"].to(DEVICE),
        }
        labels = batch["label"].float().to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Track loss
        loss_value = loss.item()
        epoch_losses.append(loss_value)
        training_log["training"]["losses"].append(loss_value)
        
        progress_bar.set_postfix({"loss": loss_value})

    # Calculate average loss for the epoch
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    training_log["training"]["avg_loss_per_epoch"].append(avg_epoch_loss)
    print(f"Average training loss: {avg_epoch_loss:.4f}")

    # Validation with IR metrics
    model.eval()
    results = {}
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            inputs = {
                "input_ids": batch["input_ids"].to(DEVICE),
                "attention_mask": batch["attention_mask"].to(DEVICE),
            }
            labels = batch["label"].cpu().numpy()
            query_ids = batch["query_id"].cpu().numpy()
            
            outputs = model(**inputs)
            scores = torch.sigmoid(outputs.logits).cpu().numpy()
            
            for qid, score, label in zip(query_ids, scores, labels):
                qid_key = int(qid) if isinstance(qid, np.ndarray) else qid
                if qid_key not in results:
                    results[qid_key] = ([], [])
                results[qid_key][0].append(score.item())
                results[qid_key][1].append(label)

    # Calculate metrics
    mrr = calculate_mrr(results)
    ndcg = calculate_ndcg(results)
    training_log["validation"]["mrr"].append(mrr)
    training_log["validation"]["ndcg"].append(ndcg)
    print(f"Validation MRR: {mrr:.4f}, NDCG@10: {ndcg:.4f}")

    # Save best model
    if mrr > best_mrr:
        best_mrr = mrr
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Saved new best model with MRR {mrr:.4f}")
    
    # Save training log after each epoch
    with open(os.path.join(OUTPUT_DIR, loss_log_file), 'w') as f:
        json.dump(training_log, f, indent=2)

print(f"Training complete. Log saved to: {os.path.join(OUTPUT_DIR, loss_log_file)}")

# Enhanced scoring function with confidence
def rerank_with_confidence(query, passages, model, tokenizer, device=DEVICE, temperature=0.3):
    model.eval()
    
    with torch.no_grad():
        # Process all passages in a batch for efficiency and consistent comparison
        inputs = tokenizer(
            [query] * len(passages),
            passages,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**inputs)
        # Apply temperature scaling to amplify differences
        logits = outputs.logits.squeeze() / temperature
        
        # Print raw scores for debugging
        print(f"Raw logits: {logits.cpu().numpy()}")
        
        confidences = torch.sigmoid(logits).cpu().numpy()
        print(f"Confidences after temperature scaling: {confidences}")
        
        # Convert to 1-100 scale
        confidence_scores = [max(1, min(100, int(round(conf * 100)))) for conf in confidences]
    
    return sorted(zip(passages, confidence_scores), key=lambda x: x[1], reverse=True)
# Example usage
query = "What causes seasons on Earth?"
passages = [
    "The Earth's tilt causes seasonal changes.",
    "Seasonal weather patterns vary by region.",
    "Earth's elliptical orbit around the Sun creates seasons.",
    "The axial tilt of 23.5 degrees results in different climate zones."
]

reranked = rerank_with_confidence(query, passages, model, tokenizer)
print("\nReranking results with confidence scores:")
for idx, (passage, score) in enumerate(reranked):
    print(f"{idx + 1}. [{score}/100] {passage}")