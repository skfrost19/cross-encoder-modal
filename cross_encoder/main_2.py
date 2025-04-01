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
from sklearn.metrics import ndcg_score
from lion_pytorch import Lion  # Requires: pip install lion-pytorch

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a cross-encoder model for document ranking')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and evaluation')
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

# Print configuration
print(f"Using configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Max Length: {MAX_LENGTH}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Device: {DEVICE}")
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

def flatten_dataset(examples):
    new_examples = {"query": [], "passage": [], "label": []}
    for i in range(len(examples["query"])):
        # Check if passages exists and has the expected structure
        if "passages" in examples and "passage_text" in examples["passages"][i] and "is_selected" in examples["passages"][i]:
            passage_texts = examples["passages"][i]["passage_text"]
            is_selected_values = examples["passages"][i]["is_selected"]
            
            for j in range(len(passage_texts)):
                new_examples["query"].append(examples["query"][i])
                new_examples["passage"].append(passage_texts[j])
                new_examples["label"].append(is_selected_values[j])
    
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

columns = ["input_ids", "attention_mask", "label"]
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

# Scheduler
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

loss_fn = torch.nn.BCEWithLogitsLoss()

# Training loop with metrics
best_mrr = 0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    
    # Training
    model.train()
    progress_bar = tqdm(train_loader, desc="Training")
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
        
        progress_bar.set_postfix({"loss": loss.item()})

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
                if qid not in results:
                    results[qid] = ([], [])
                results[qid][0].append(score.item())
                results[qid][1].append(label)

    # Calculate metrics
    mrr = calculate_mrr(results)
    ndcg = calculate_ndcg(results)
    print(f"Validation MRR: {mrr:.4f}, NDCG@10: {ndcg:.4f}")

    # Save best model
    if mrr > best_mrr:
        best_mrr = mrr
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Saved new best model with MRR {mrr:.4f}")

# Enhanced scoring function with confidence
def rerank_with_confidence(query, passages, model, tokenizer, device=DEVICE):
    model.eval()
    scores = []
    
    with torch.no_grad():
        for passage in passages:
            inputs = tokenizer(
                query,
                passage,
                padding="max_length",
                truncation="only_second",
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            confidence = torch.sigmoid(outputs.logits).item()
            # Convert to 1-100 scale with clipping
            confidence_score = max(1, min(100, int(round(confidence * 100))))
            scores.append(confidence_score)
    
    return sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

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