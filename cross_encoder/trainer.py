import os
import logging
import traceback

import torch
from datasets import load_dataset, load_from_disk

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

from lion_pytorch import Lion  # Requires: pip install lion-pytorch

# Use the mounted volume path for all persistent storage
MODEL_SAVE_PATH = "/root/longformer_crossencoder/all_mini_LM"

# Create paths for dataset storage within the mounted volume
DATASET_DIR = "/root/longformer_crossencoder/datasets"
TRAIN_DATASET_PATH = os.path.join(DATASET_DIR, "ms-marco-train")
EVAL_DATASET_PATH = os.path.join(DATASET_DIR, "ms-marco-eval")

def main():
    # Create the dataset directory if it doesn't exist
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    model_name = "microsoft/MiniLM-L12-H384-uncased"

    # Set the log level to INFO to get more information
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    train_batch_size = 256
    num_epochs = 1
    dataset_size = 2_000_000

    # 1. Define our CrossEncoder model
    # Set the seed so the new classifier weights are identical in subsequent runs
    torch.manual_seed(12)
    model = CrossEncoder(model_name)
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/sentence-transformers/msmarco
    logging.info("Read train dataset")
    try:
        train_dataset = load_from_disk(TRAIN_DATASET_PATH)  # Updated path
        eval_dataset = load_from_disk(EVAL_DATASET_PATH)    # Updated path
    except FileNotFoundError:
        logging.info("The dataset has not been fully stored as texts on disk yet. We will do this now.")
        corpus = load_dataset("sentence-transformers/msmarco", "corpus", split="train")
        corpus = dict(zip(corpus["passage_id"], corpus["passage"]))
        queries = load_dataset("sentence-transformers/msmarco", "queries", split="train")
        queries = dict(zip(queries["query_id"], queries["query"]))
        dataset = load_dataset("sentence-transformers/msmarco", "triplets", split="train")
        dataset = dataset.select(range(dataset_size // 2))

        def id_to_text_map(batch):
            return {
                "query": [queries[qid] for qid in batch["query_id"]] * 2,
                "passage": [corpus[pid] for pid in batch["positive_id"]]
                + [corpus[pid] for pid in batch["negative_id"]],
                "score": [1.0] * len(batch["positive_id"]) + [0.0] * len(batch["negative_id"]),
            }

        dataset = dataset.map(id_to_text_map, batched=True, remove_columns=dataset.column_names)
        dataset = dataset.train_test_split(test_size=10_000)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # Save to persistent storage in the mounted volume
        train_dataset.save_to_disk(TRAIN_DATASET_PATH)  # Updated path
        eval_dataset.save_to_disk(EVAL_DATASET_PATH)    # Updated path
        
        logging.info(
            f"The dataset has now been stored in the mounted volume at {DATASET_DIR}. "
            "The script will now stop to ensure that memory is freed. "
            "Please restart the script to start training."
        )
        quit()
    logging.info(train_dataset)

    # 3. Define our training loss
    loss = BinaryCrossEntropyLoss(model)

    # 4. Define the evaluator. We use the CrossEncoderNanoBEIREvaluator, which is a light-weight evaluator for English reranking
    evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=train_batch_size)
    evaluator(model)

        # 6. Create the trainer & start training
    
    

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-{short_model_name}-msmarco-bce"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"{MODEL_SAVE_PATH}/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=10000,
        save_strategy="steps",
        save_steps=10000,
        save_total_limit=2,
        logging_steps=4000,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
        dataloader_num_workers=4,
    )

    # Setup Lion optimizer
    optimizer = Lion(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.99)  # Default Lion betas
    )

    # 6. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        optimizers = (optimizer, None),  # Pass the optimizer and scheduler to the trainer
    )
    trainer.train()

    # 7. Evaluate the final model, useful to include these in the model card
    evaluator(model)

    # 8. Save the final model
    final_output_dir = f"{MODEL_SAVE_PATH}/{run_name}/final"
    model.save_pretrained(final_output_dir)

    # 9. (Optional) save the model to the Hugging Face Hub!
    # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
            f"`huggingface-cli login`, followed by loading the model using `model = CrossEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )

def push_model_to_hub(model_path, model_name, run_name=None, org=None, private=False):
    """
    Pushes a trained CrossEncoder model to the Hugging Face Hub with a detailed README.
    
    Args:
        model_path (str): Path to the saved model
        model_name (str): Original base model name
        run_name (str, optional): Name for the model on HF Hub. Defaults to None.
        org (str, optional): Organization name for upload. Defaults to None.
        private (bool, optional): Whether to make the model private. Defaults to False.
    
    Returns:
        bool: True if successful, False otherwise
    """
    import os
    import logging
    import traceback
    from sentence_transformers.cross_encoder import CrossEncoder
    
    # Set up logging
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    
    try:
        # Load the model
        logging.info(f"Loading model from {model_path}")
        model = CrossEncoder(model_path)
        
        # Get short model name for display
        short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
        
        # Set the repo name
        if run_name is None:
            run_name = f"reranker-{short_model_name}-msmarco-bce"
        
        repo_name = run_name
        if org:
            repo_name = f"{org}/{run_name}"
        
        # Create README content
        readme_content = f"""
# {run_name}

This is a cross-encoder model fine-tuned for passage ranking and information retrieval.

## Model Description

- **Model type:** Cross-Encoder for retrieval/ranking
- **Base model:** {model_name}
- **Training data:** MS MARCO Passage Ranking dataset
- **Task:** Passage ranking, document reranking, information retrieval

## Usage

This model is a cross-encoder that takes a query and a passage as input and returns a relevance score between 0 and 1.

```python
from sentence_transformers.cross_encoder import CrossEncoder

model = CrossEncoder('{repo_name}')

# Single example
query = "What is the capital of France?"
passage = "Paris is the capital and most populous city of France."
score = model.predict([(query, passage)])
print(f"Relevance score: {{score[0]:.4f}}")

# Multiple examples for reranking
passages = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital and largest city of Germany.",
    "France is in Europe and its capital is Paris.",
    "The Eiffel Tower is located in Paris, France."
]

# Compute similarities between the query and all passages
scores = model.predict([(query, passage) for passage in passages])

# Sort passages by decreasing similarity scores
passage_scores = list(zip(passages, scores))
passage_scores = sorted(passage_scores, key=lambda x: x[1], reverse=True)

# Print the passages with their scores
for passage, score in passage_scores:
    print(f"{{score:.4f}} - {{passage}}")
"""
        # Save the README to the model directory
        readme_path = os.path.join(model_path, "README.md")
        with open(readme_path, "w") as readme_file:
            readme_file.write(readme_content)
        logging.info(f"README saved to {readme_path}")
        # Push to the Hugging Face Hub
        model.push_to_hub(repo_name, private=private)
        logging.info(f"Model pushed to Hugging Face Hub at {repo_name}")
        return True
    except Exception as e:
        logging.error(f"Error pushing model to Hugging Face Hub:\n{traceback.format_exc()}")
        return False
    finally:
        # Clean up the README file
        if os.path.exists(readme_path):
            os.remove(readme_path)
            logging.info(f"Removed temporary README file at {readme_path}")
        else:
            logging.warning(f"README file not found at {readme_path} for cleanup.")
if __name__ == "__main__":
    # main()
    # Example usage of pushing the model to the Hugging Face Hub
    push_model_to_hub(
        model_path=f"{MODEL_SAVE_PATH}/reranker-MiniLM-L12-H384-uncased-msmarco-bce/final",
        model_name="microsoft/MiniLM-L12-H384-uncased",
        run_name="reranker-MiniLM-L12-H384-uncased-msmarco-bce",
        org=None,
        private=False
    )

    # training logs:- https://modal.com/apps/model-merging/main/ap-U94j6dVXx260A9IBXF9IJR?start=1743441282.824&end=1743527682.824&live=true&activeTab=logs