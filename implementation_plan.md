# Implementation Plan: Learnable Causal Sensitivity \(\gamma(Q)\)

We propose replacing the fixed causal penalty hyperparameter \(\gamma\) with a learnable, question-conditioned causal scale \(\gamma(Q)\). This makes the model's sensitivity to visual interventions adapt dynamically to the semantic context of the question.

---

## Technical Design

### CQCNet Integration
In [`models/cqc_net.py`](file:///Users/fs525/Desktop/CQC2/models/cqc_net.py), we will add a **Causal Projection Head**:
```python
self.causal_gamma_head = nn.Sequential(
    nn.Linear(text_dim, text_dim // 2),
    nn.ReLU(),
    nn.Linear(text_dim // 2, 1),
    nn.Softplus() # Ensures gamma is always positive
)
```
During the forward pass, this head will map the question features to a dynamic scalar:
\[ \gamma_i = \text{causal\_gamma\_head}(q\_feat_i) \]

### Causal Contrastive Decoder Adjustment
The `CausalContrastiveDecoder` in [`models/causal_decoder.py`](file:///Users/fs525/Desktop/CQC2/models/causal_decoder.py) will receive this tensor \(\gamma\) in its forward call and apply it element-wise:
```python
# Rather than using a static self.gamma, we use the input tensor gamma
hallucination_score = torch.sigmoid(-gamma * ice)
```

---

## Proposed Changes

### [MODIFY] [cqc_net.py](file:///Users/fs525/Desktop/CQC2/models/cqc_net.py)
*   Add the `causal_gamma_head` MLP.
*   Return both VQA logits and the computed question-conditioned `gamma` in the forward pass dict.

### [MODIFY] [causal_decoder.py](file:///Users/fs525/Desktop/CQC2/models/causal_decoder.py)
*   Adjust `forward` and `calibrate_generative_logits` to take `gamma` as a tensor argument instead of using `self.gamma`.

### [MODIFY] [evaluate_causal_vqa.py](file:///Users/fs525/Desktop/CQC2/scripts/evaluate_causal_vqa.py) & [benchmark_comparison.py](file:///Users/fs525/Desktop/CQC2/scripts/benchmark_comparison.py)
*   Pass the dynamic `gamma` outputted by `CQCNet` to the `CausalContrastiveDecoder`.

---

## Verification Plan

### Automated Tests
*   `python scripts/verify_pipeline.py` to verify the pipeline compiles and executes with the learnable causal parameter.
*   `python scripts/benchmark_comparison.py` to confirm that the calibrated outputs are computed correctly.
