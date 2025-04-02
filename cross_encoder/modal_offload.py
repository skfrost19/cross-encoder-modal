import modal

modal_volume = modal.Volume.from_name("longformer_crossencoder", create_if_missing=True)

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(  # required to build flash-attn
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "lion-pytorch",
        "numpy",
        "scikit-learn",
        "hf_xet",
        "tiktoken",
        "sentence-transformers",
        "wandb",
        "accelerate>=0.26.0",
    )
).add_local_dir("D:\\Thesis\\4th Sem\\cross_encoder", remote_path="/root/cross_encoder")


MOUNT_DIR = "/root/longformer_crossencoder"
MODEL_PATH = "/root/longformer_crossencoder/deberta_crossencoder"

app = modal.App(
    name="cross-encoder",
    image=image,
    volumes={MOUNT_DIR: modal_volume},
)


@app.function(
    gpu="A100-80GB:1",
    timeout=72000,
    secrets=[modal.Secret.from_name("my-huggingface-secret"), modal.Secret.from_name("my-wandb-secret")],
)
def evaluate():
    import os

    os.system(f"python cross_encoder/main.py --batch_size 40 --epochs 3 --lr 2e-5 --max_length 1024 --model_name microsoft/deberta-v3-base --output_dir {MODEL_PATH}")
    

@app.function(
    gpu="A100-80GB:1",
    timeout=72000,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
def infer_model():
    import os

    os.system(f'python cross_encoder/inference.py --model_path {MODEL_PATH} --query "What causes seasons on Earth?" --temperature 0.7')

@app.function(
    gpu="A100-80GB:1",
    timeout=72000,
    secrets=[modal.Secret.from_name("my-huggingface-secret"), modal.Secret.from_name("my-wandb-secret")],
)
def trainer():
    import os

    os.system(f"python cross_encoder/trainer.py")
    
@app.function(
    gpu="A100-80GB:1",
    timeout=7200,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
def validate():
    import os
    
    MODEL_PATH = "/root/longformer_crossencoder/all_mini_LM/reranker-MiniLM-L12-H384-uncased-msmarco-bce/final"
    
    os.system(f"python cross_encoder/validate_model.py --model_path {MODEL_PATH} --batch_size 512 --num_samples 2000")

@app.function(
    gpu="A100-80GB:1",
    timeout=7200,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
def validate_separate():
    import os
    
    MODEL_PATH = "/root/longformer_crossencoder/all_mini_LM/reranker-MiniLM-L12-H384-uncased-msmarco-bce/final"
    EVAL_DATASET_PATH = "/root/longformer_crossencoder/datasets/ms-marco-eval"
    OUTPUT_PATH = "/root/longformer_crossencoder/validation_results.json"
    
    os.system(f"python cross_encoder/validate_s_marco.py --model_path {MODEL_PATH} --eval_dataset {EVAL_DATASET_PATH} --output_path {OUTPUT_PATH} --batch_size 512")

@app.local_entrypoint()
def main():
    """
    Main entry point for the evaluation script.

    This function defines a list of pretrained models to be evaluated and calls the evaluate function
    to perform the evaluation.

    Args:
        None
    """
    # download_models_huggingface.remote()
    # infer_model.remote()
    # evaluate.remote()
    # trainer.remote()
    # validate.remote()
    validate_separate.remote()


# To run this:
# - Make sure you have modal api-key configured and huggingface token configured in modal secrets.
# - Run `modal run evaluation.py`
# - You can check the results in the `evaluation-results` volume
# - If you want to download the results, you can use the `modal volume get evaluation-results /` command and it will download the entire voulme content to your local machine.