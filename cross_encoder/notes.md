# Model Performance Comparison

## [gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)

| Metric      | Epoch 1 | Epoch 2 | Epoch 3 |
|-------------|---------|---------|---------|
| P_10        | 0.7884  | 0.7721  | 0.7488  |
| map         | 0.4794  | 0.4577  | 0.4521  |
| ndcg        | 0.6865  | 0.6772  | 0.6674  |
| ndcg_cut_5  | 0.7229  | 0.7008  | 0.6801  |
| recall_5    | 0.1002  | 0.0973  | 0.0929  |
| recip_rank  | 0.9612  | 0.9477  | 0.9399  |

## [MiniLM-L12-H384-uncased](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)

| Metric      | Epoch 1 | Epoch 2 |
|-------------|---------|---------|
| P_10        | 0.8140  | 0.8070  |
| map         | 0.4858  | 0.4755  |
| ndcg        | 0.6841  | 0.6790  |
| ndcg_cut_5  | 0.7163  | 0.7099  |
| recall_5    | 0.1022  | 0.1025  |
| recip_rank  | 0.9457  | 0.9496  |