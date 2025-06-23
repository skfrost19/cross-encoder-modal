# Comparative Analysis of Lion and AdamW Optimizers for Cross-Encoder Reranking

**Shahil Kumar**  
*IIIT Allahabad*

---

## Abstract

Modern information retrieval systems employ two-stage pipelines: efficient retrieval followed by computationally intensive reranking. This study investigates the impact of the **Lion optimizer** compared to **AdamW** for fine-tuning cross-encoder rerankers across three transformer models.

### Key Findings
- **ModernBERT + Lion** achieved highest performance (NDCG@10: 0.7225)
- **Lion optimizer** shows 2.67% to 10.33% GPU efficiency gains
- **Optimizer choice** significantly interacts with model architecture and hyperparameters

---

## Introduction

### Two-Stage Retrieval Pipeline
1. **First Stage**: Efficient retrieval (BM25, dense vectors)
   - Retrieves hundreds/thousands of candidates
2. **Second Stage**: Sophisticated reranking
   - Cross-encoders for higher precision at top ranks

### Cross-Encoder Architecture
- **Input**: `[CLS] query [SEP] document [SEP]`
- **Deep interaction**: Token-level query-document modeling
- **Output**: Relevance score via sigmoid activation
- **Superior accuracy** but higher computational cost

---

## Research Motivation

### Why Study Optimizers for Cross-Encoders?
- **Optimizer choice** significantly impacts performance and training efficiency
- **Lion optimizer** recently proposed with claimed improvements over AdamW
- **Limited evaluation** in NLP/IR domain (mostly vision tasks)

### Research Questions
1. How does Lion compare to AdamW for cross-encoder training?
2. Does the choice interact with different model architectures?
3. What are the practical efficiency implications?

---

## Methodology

### Models Evaluated
| Model | Architecture | Context Length | Key Features |
|-------|-------------|----------------|--------------|
| **MiniLM-L12-H384** | Distilled BERT | 512 tokens | Smaller, faster |
| **GTE-multilingual-base** | Multilingual transformer | 8192 tokens | Long context support |
| **ModernBERT-base** | Enhanced BERT | 8192 tokens | RoPE, Flash Attention, GeGLU |

### Optimizers Compared
- **AdamW**: `weight_decay=0.01`, varying learning rates
- **Lion**: `betas=(0.9, 0.99)`, `weight_decay=0.01`, simplified update rule

### Training Configuration
- **Dataset**: MS MARCO passage ranking (~2M pairs)
- **Batch Size**: 64
- **Epochs**: 3
- **Infrastructure**: 3x NVIDIA L40S-48GB GPUs via Modal

---

## Experimental Setup

### Training Parameters
| Model | Optimizer | Learning Rate | Scheduler |
|-------|-----------|---------------|-----------|
| MiniLM | Both | 2e-5 | None |
| GTE | Both | 2e-5 | None |
| ModernBERT | AdamW | 2e-5 | None |
| ModernBERT | Lion | 2e-6 | CosineAnnealing |

### Evaluation Datasets
- **TREC DL 2019**: 43 queries with graded relevance
- **MS MARCO Dev**: Larger query set for MRR@10

### Metrics
- NDCG@10, MAP, Recall@10, R-Prec, P@10 (TREC DL)
- MRR@10 (MS MARCO Dev)

---

## Key Results: Performance Comparison

### Best Performance by Model-Optimizer Combination

| Model + Optimizer | NDCG@10 | MAP | MRR@10 | Best Epoch |
|-------------------|---------|-----|--------|------------|
| **ModernBERT + Lion** | **0.7225** | **0.5121** | 0.5988 | 2 |
| **GTE + AdamW** | 0.7224 | 0.5005 | 0.5942 | 1 |
| **MiniLM + AdamW** | 0.7127 | 0.4908 | 0.5826 | 3 |
| **MiniLM + Lion** | 0.7031 | 0.4858 | **0.5988** | 1/3 |

### Key Observations
- **ModernBERT + Lion** achieved overall best TREC DL performance
- **Performance varies by epoch** - not always monotonic improvement
- **Different optimizers peak at different epochs**

---

## GPU Efficiency Analysis

### Resource Utilization Comparison

| Model | Optimizer | Mean Usage (%) | Peak Usage (%) | Efficiency Gain |
|-------|-----------|----------------|----------------|-----------------|
| MiniLM | AdamW | 33.09 | 35.25 | - |
| MiniLM | Lion | 32.21 | 34.64 | **+2.67%** |
| GTE | AdamW | 73.04 | 78.08 | - |
| GTE | Lion | 65.50 | 69.69 | **+10.33%** |
| ModernBERT | AdamW | 77.04 | 81.40 | - |
| ModernBERT | Lion | 74.35 | 79.45 | **+3.49%** |

### Efficiency Insights
- **Lion consistently more efficient** across all models
- **Largest gain with GTE** (10.33% improvement)
- **Practical computational savings** during training

---

## Model-Specific Analysis

### MiniLM Performance
- **AdamW**: Higher peak TREC performance, stable over epochs
- **Lion**: Better MS MARCO MRR, but TREC performance declines after epoch 1
- **Conclusion**: AdamW more stable for longer training

### GTE Performance  
- **AdamW clearly superior** across most metrics
- **Peak performance early** (Epoch 1-2), then degrades
- **Lion shows weaker overall performance**

### ModernBERT Performance
- **Lion significantly outperforms AdamW** with tailored hyperparameters
- **Specific training regime**: Low LR (2e-6) + Cosine Annealing
- **Best overall results** achieved with this combination

---

## State-of-the-Art Comparison

### TREC DL 2019 NDCG@10 Performance

| System Type | Best Model | NDCG@10 |
|-------------|------------|---------|
| **Our Work** | **ModernBERT + Lion** | **0.7225** |
| Hybrid Models | Bi-encoder + doc2query-T5 | 0.719 |
| Cross-Encoders | ms-marco-electra-base | 0.719 |
| Dense Retrieval | BM25 Neg | 0.664 |
| Single-stage | doc2query-T5 | 0.642 |

### Achievement
- **State-of-the-art performance** on TREC DL 2019
- **Surpassed existing benchmarks** with appropriate optimizer choice
- **Competitive across multiple model sizes**

---

## Training Dynamics Insights

### Performance Trends Over Epochs
- **Not monotonic**: Performance doesn't always improve with more training
- **Model-dependent**: Different models peak at different epochs
- **Optimizer interaction**: Lion vs AdamW show different convergence patterns

### Hyperparameter Sensitivity
- **ModernBERT + Lion**: Requires specific LR (2e-6) and scheduler
- **GTE + AdamW**: Works well with standard settings (2e-5, no scheduler)
- **MiniLM**: Relatively stable across both optimizers

### Practical Implications
- **Checkpoint selection crucial** based on validation performance
- **Hyperparameter tuning** essential for Lion optimizer
- **Early stopping** may be beneficial for some combinations

---

## Technical Advantages of Lion

### Optimizer Characteristics
- **Simpler update rule**: `update = sign(momentum) * lr`
- **Lower memory usage**: No second moment estimates
- **Momentum-based**: Only tracks first moment

### Practical Benefits
1. **Memory efficiency**: Reduced GPU memory requirements
2. **Computational efficiency**: Simpler mathematical operations
3. **Training speed**: Faster per-iteration processing

### When Lion Excels
- **Larger models**: Greater efficiency gains (GTE: 10.33%)
- **Specific architectures**: ModernBERT with tailored hyperparameters
- **Resource-constrained environments**: Better GPU utilization

---

## Limitations and Future Work

### Current Limitations
- **Limited hyperparameter exploration** for Lion across all models
- **Single dataset focus**: MS MARCO for training
- **Short context evaluation**: Limited long-document analysis

### Future Research Directions
1. **Comprehensive hyperparameter search** for Lion across all models
2. **Long-context evaluation**: Leverage 8K token capabilities of GTE/ModernBERT
3. **Memory usage analysis**: Detailed comparison of optimizer memory requirements
4. **Broader dataset evaluation**: Multiple IR benchmarks
5. **Hardware scaling study**: Different GPU configurations and model scales

---

## Practical Recommendations

### For Practitioners

#### When to Use Lion:
- **ModernBERT-based models** with low LR + Cosine Annealing
- **Resource-constrained training** environments
- **Large-scale model training** where efficiency matters

#### When to Use AdamW:
- **GTE-based models** with standard hyperparameters
- **Stable, predictable training** requirements
- **Well-established training pipelines**

### Implementation Tips
1. **Start with model-specific configurations** from this study
2. **Monitor multiple epochs** - don't assume monotonic improvement
3. **Consider efficiency gains** when scaling to larger deployments
4. **Validate on multiple metrics** and datasets

---

## Conclusions

### Key Findings Summary
1. **Optimizer choice matters**: Significant interaction with model architecture
2. **Lion achieves SOTA**: Best TREC DL 2019 performance with ModernBERT
3. **Efficiency gains**: 2.67-10.33% GPU utilization improvements
4. **Context-dependent**: No universally superior optimizer

### Impact
- **Demonstrates Lion's potential** in NLP/IR domain
- **Provides practical guidance** for optimizer selection
- **Achieves competitive results** with improved efficiency

### Final Takeaway
The choice between Lion and AdamW should be **model-specific and empirically validated**, with Lion showing particular promise for modern architectures when properly tuned.

---

## Acknowledgments

- **Modal Labs** for cloud computing platform and GPU resources
- **3x NVIDIA L40S-48GB GPUs** for experimental infrastructure
- **Open source community** for libraries and benchmarks used

### Resources
- **Code**: https://github.com/skfrost19/Cross-Encoder-Lion-vs-AdamW
- **Models**: https://huggingface.co/collections/skfrost19/rerenkers-681320776cfb45e44b18f5f1

---

## Questions & Discussion

**Thank you for your attention!**

*Research demonstrates that optimizer choice significantly impacts both performance and efficiency in cross-encoder reranking tasks.*