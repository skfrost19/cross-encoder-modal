import ir_datasets
import pandas as pd
from sentence_transformers import CrossEncoder
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import subprocess
import os

# Configuration
MODEL_NAME = "skfrost19/reranker-gte-multilingual-base-msmarco-bce-ep-2"
BM25_RUN_PATH = "runs/run.bm25.trec"
RERANKED_RUN_PATH = "runs/reranked.trec"
TREC_EVAL_METRICS = ["ndcg_cut.10", "map", "recall.1000"]

# Load dataset
print("Loading TREC DL 2019 dataset...")
# First load the base MS MARCO passage dataset
msmarco = ir_datasets.load("msmarco-passage")
dataset = ir_datasets.load("msmarco-passage/trec-dl-2019")
queries = {q.query_id: q.text for q in dataset.queries_iter()}

# Create output directory
os.makedirs(os.path.dirname(BM25_RUN_PATH), exist_ok=True)

# Run BM25 retrieval if needed
# Initialize BM25 searcher
print("Loading BM25 index...")
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
if searcher is None:
    print("Error loading prebuilt index 'msmarco-v1-passage'")
    exit()

# Run BM25 retrieval
print("Running BM25 retrieval...")
bm25_results = []
for qid in tqdm(queries, desc="Processing queries"):
    query_text = queries[qid]
    hits = searcher.search(query_text, k=1000)
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
bm25_df.to_csv(BM25_RUN_PATH, sep='\t', header=False, index=False)
print(f"BM25 results saved to {BM25_RUN_PATH}")

# Load passage texts
print("Loading MS MARCO passage store...")
docs = ir_datasets.load("msmarco-passage").docs_store()

# Initialize model
print(f"Loading model {MODEL_NAME}...")
model = CrossEncoder(MODEL_NAME, trust_remote_code=True, device="cuda:0")

# Rerank results
print("Reranking with cross-encoder...")
with open(RERANKED_RUN_PATH, "w") as f_out:
    for qid in tqdm(queries, desc="Processing queries"):
        query_text = queries[qid]

        # Get BM25 docs for current query
        query_docs_df = bm25_df[bm25_df["qid"] == qid]
        query_docs = query_docs_df.sort_values("rank")["doc_id"].tolist()[:1000]

        # Retrieve document texts
        passages = []
        valid_doc_ids = []
        for doc_id in query_docs:
            doc = docs.get(doc_id)
            if doc is not None:
                passages.append(doc.text)
                valid_doc_ids.append(doc_id)

        if not passages:
            continue

        # Score documents
        pairs = [(query_text, passage) for passage in passages]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)

        # Sort documents by score
        sorted_docs = sorted(
            zip(valid_doc_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Write TREC-formatted results
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            line = f"{qid} Q0 {doc_id} {rank} {score} {MODEL_NAME}\n"
            f_out.write(line)

# Install trec_eval if not exists
if not os.path.exists("trec_eval"):
    print("Installing trec_eval...")
    subprocess.run(["git", "clone", "https://github.com/usnistgov/trec_eval.git"], check=True)
    subprocess.run("cd trec_eval && make", shell=True, check=True)


# Run trec_eval
print("\nEvaluation results:")
qrels_path = dataset.qrels_path()
subprocess.run([
    "./trec_eval/trec_eval",
    *[arg for metric in TREC_EVAL_METRICS for arg in ("-m", metric)],
    qrels_path,
    RERANKED_RUN_PATH
])