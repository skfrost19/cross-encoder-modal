import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_cross_encoder(model_path, device=None):
    """Load a saved cross-encoder model and tokenizer"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loaded cross-encoder model from {model_path} on {device}")
    return model, tokenizer, device

def rerank_with_confidence(query, passages, model_path=None, model=None, tokenizer=None, 
                          max_length=1024, batch_size=8, temperature=1.0):
    """
    Rerank passages using a cross-encoder model with confidence scores
    
    Args:
        query (str): The search query
        passages (list): List of passage texts to rerank
        model_path (str): Path to saved model (ignored if model and tokenizer provided)
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        max_length (int): Maximum sequence length
        batch_size (int): Batch size for processing
        temperature (float): Temperature for scaling confidence scores
        
    Returns:
        list: Ranked passages with their scores [(passage, raw_score, display_score), ...]
    """
    # Load model if not provided
    if model is None or tokenizer is None:
        if model_path is None:
            raise ValueError("Either provide model_path or both model and tokenizer")
        model, tokenizer, device = load_cross_encoder(model_path)
    else:
        device = next(model.parameters()).device
    
    model.eval()
    results = []
    
    # Process in batches for efficiency
    for i in range(0, len(passages), batch_size):
        batch_passages = passages[i:i+batch_size]
        batch_queries = [query] * len(batch_passages)
        
        # Tokenize
        inputs = tokenizer(
            batch_queries,
            batch_passages,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"Model outputs: {outputs.logits.squeeze().cpu().numpy()}")
            scores = torch.sigmoid(outputs.logits / temperature).squeeze().cpu().numpy()
            
            # Handle single result case
            if len(batch_passages) == 1:
                scores = [scores.item()]
            
            # Store results
            for passage, score in zip(batch_passages, scores):
                # Raw score
                raw_score = float(score)
                # Display score (1-100 scale)
                display_score = max(1, min(100, int(round(raw_score * 100))))
                results.append((passage, raw_score, display_score))
    
    # Sort by score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rerank passages using a cross-encoder model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for scaling confidence")
    args = parser.parse_args()
    
    # Example passages
    passages = [
        "The Earth's tilt causes seasonal changes.",
        "Seasonal weather patterns vary by region.",
        "Earth's elliptical orbit around the Sun creates seasons.",
        "The axial tilt of 23.5 degrees results in different climate zones.",
        "The 23.5° tilt of Earth's axis causes the seasons because it alters the intensity of sunlight regions receive throughout the year."
    ]
    
    # Rerank
    reranked = rerank_with_confidence(
        args.query, 
        passages,
        model_path=args.model_path,
        temperature=args.temperature
    )
    
    # Print results
    print(f"\nQuery: {args.query}")
    print("\nReranking results with confidence scores:")
    for idx, (passage, raw_score, display_score) in enumerate(reranked):
        print(f"{idx + 1}. [{display_score}/100] ({raw_score:.4f}) {passage}")