# Architecture Documentation

**Paper:** Hybrid Neural–Heuristic TSP Solver with Attention-Guided Refinement for Real-Time AMR Coordination  
**Authors:** BLIND REVIEW 
**Institution:** BLIND REVIEW 

---

## Overview

This document provides implementation details for the hybrid neural-heuristic architecture described in the paper. For complete algorithmic specifications, see **Appendix D** (Algorithms 1-2) in the paper.

---

## 1. Transformer Encoder-Decoder

### 1.1 Encoder Specification

**Architecture** (Section 4.2):
- **Layers (L):** 6
- **Attention Heads (H):** 8
- **Embedding Dimension (d):** 256
- **Feed-Forward Dimension:** 1024 (4×d)
- **Activation:** ReLU
- **Normalization:** Layer normalization (pre-norm)
- **Dropout:** 0.1 (training only)

**Input Processing:**
```
Coordinates (x, y) → FFN → d-dimensional embedding h⁽⁰⁾
```

**Layer Structure (Eq. 10-11):**
```
hₑ⁽ℓ⁾ = LayerNorm(h⁽ℓ⁻¹⁾ + MHA(h⁽ℓ⁻¹⁾))
h⁽ℓ⁾  = LayerNorm(hₑ⁽ℓ⁾ + FFN(hₑ⁽ℓ⁾))
```

**Multi-Head Attention (MHA):**
- Scaled dot-product attention (Eq. 12)
- 8 parallel heads
- Head dimension: d/H = 32
- Final attention matrix A ∈ ℝ^(N×N) averaged across heads

### 1.2 Decoder Specification

**Type:** Autoregressive pointer network

**Mechanism (Eq. 13-14):**
```
eₜ,ᵢ = qₜᵀkᵢ / √dₖ    (compatibility scores)
p(i|t) = softmax(eₜ,ᵢ)  (probability distribution)
```

**Decoding:**
- Greedy selection: aₜ = argmax p(i|t)
- Masking: only unvisited nodes Uₜ considered
- Query vector qₜ: combination of last selected node + pooled context

---

## 2. Attention-Guided 2-Opt Refinement

### 2.1 Candidate Selection

**Scoring Function (Eq. 17-18):**
```python
s(i, j) = A_ij + A_ji                    # Symmetric attention
score(i, j) = α·s(i,j) + (1-α)·(1 - d(i,j)/d_max)
```

**Parameters:**
- α = 0.7 (attention weight)
- K = min(3N, N(N-1)/2) (candidate set size)

**Selection (Eq. 19):**
```
C = TopK{score(i,j) : (i,j) ∈ E}
```

### 2.2 2-Opt Operator

**Improvement Check (Eq. 21):**
```
Δᵢⱼ = d(πᵢ,πⱼ) + d(πᵢ₊₁,πⱼ₊₁) - d(πᵢ,πᵢ₊₁) - d(πⱼ,πⱼ₊₁)

If Δᵢⱼ < 0: reverse segment (πᵢ₊₁,...,πⱼ)
```

**Termination:**
- No beneficial swap in C, OR
- Maximum iterations reached (2000)

**Complexity:** O(N log N) expected (Section 4.5)

---

## 3. Training Procedure

### 3.1 Dataset

**Distribution:** Uniform random in [0, 1000]²  
**Sizes:** N ∈ {20, 30, 40, 50}  
**Instances:** 40,000 training + 4,000 validation

### 3.2 Adaptive Curriculum Learning

**Supervision Targets (Eq. 15-16):**
```
π_target = {
  π_refined  with probability p(e) = p_max·min(1, e/E_warmup)
  π_greedy   otherwise
}
```

**Parameters:**
- p_max = 0.7 (refined target probability)
- E_warmup = 60 epochs (warmup period)

**Rationale:** Balances stability (greedy) with quality (refined)

### 3.3 Optimization

**Optimizer:** Adam (Eq. 23)
- Learning rate η = 3×10⁻⁴
- Warmup: 1,000 steps linear ramp
- β₁ = 0.9, β₂ = 0.999

**Loss Function (Eq. 23):**
```
L(θ) = -Σₜ log p(πₜᵗᵃʳᵍᵉᵗ | t; θ)
```

**Regularization:**
- Dropout: 0.1 (FFN layers)
- Weight decay: λ = 10⁻⁵
- Gradient clipping: norm 1.0

**Batch Size:** 128  
**Training Time:** ~8 hours on Apple M2 Pro (19-core GPU)

---

## 4. Hardware Optimization

### 4.1 GPU Implementation (Section 5.1)

**Platform:** Apple Metal Performance Shaders (MPS)

**Optimizations:**
1. **Fused Attention Kernels:** Reduce memory transfers
2. **Persistent Buffers:** Avoid repeated allocation
3. **Mixed Precision (FP16):** 2× speedup, minimal accuracy loss

**Encoder Latency (N=100):**
- Embedding: 3-5 ms
- 6 Transformer layers: 10-15 ms
- **Total:** 13-20 ms

**Decoder Latency (N=100):**
- Autoregressive steps: 15-25 ms

### 4.2 CPU Implementation

**Platform:** Numba JIT (Python → LLVM)

**Optimizations:**
1. **SIMD Vectorization:** Distance computations batched
2. **Cache Optimization:** Memory access patterns aligned
3. **In-Place Operations:** Segment reversal without copying

**2-Opt Latency (N=100):**
- Candidate scoring: 5-10 ms
- Refinement iterations: 20-40 ms
- **Total:** 25-50 ms

### 4.3 Embedded Deployment

**Platform:** NVIDIA Jetson AGX Orin

**Optimizations:**
1. **TensorRT Compilation:** INT8 quantization
2. **NEON Intrinsics:** ARM SIMD for 2-opt
3. **Thread Pinning:** CPU affinity for determinism

**Latency (N=100):**
- Encoder+Decoder: 25-35 ms
- 2-Opt: 30-45 ms
- **Total:** 55-80 ms (sub-100ms ✓)

---

## 5. Inference Pipeline

### 5.1 End-to-End Flow

```
Input: Coordinates (N×2 array)
  ↓
[GPU] Embedding + Encoding (10-15ms)
  ↓
[GPU] Attention Extraction (1-2ms)
  ↓
[GPU] Greedy Decoding (15-25ms)
  ↓
[CPU] Candidate Selection (5-10ms)
  ↓
[CPU] 2-Opt Refinement (20-40ms)
  ↓
Output: Tour π (permutation)
```

**Total Latency:** 51-92ms for N=100 (median 66ms)

### 5.2 Real-Time Guarantees (Section 5.3)

| N   | Median | 95th pct | RT Capable |
|-----|--------|----------|------------|
| 50  | 41 ms  | 58 ms    | ✓ (Hard)   |
| 100 | 66 ms  | 89 ms    | ✓ (Hard)   |
| 150 | 101 ms | 127 ms   | ∼ (95%)    |
| 200 | 154 ms | 198 ms   | × (Soft)   |

**Hard RT:** 100% instances < 100ms deadline  
**Soft RT:** >90% instances < 100ms deadline

---

## 6. Model Checkpoints

### 6.1 Pretrained Weights

**Location:** `models/transformer_encoder.pth` (upon release)

**Structure:**
```python
{
  'encoder_state_dict': ...,  # 6 Transformer layers
  'decoder_state_dict': ...,  # Pointer decoder
  'hyperparameters': {
    'L': 6, 'H': 8, 'd': 256,
    'p_max': 0.7, 'alpha': 0.7
  }
}
```

**Loading:**
```python
import torch
checkpoint = torch.load('models/transformer_encoder.pth')
model.load_state_dict(checkpoint['encoder_state_dict'])
```

### 6.2 Quantized Models (Embedded)

**INT8 Version:** `models/transformer_int8.trt` (TensorRT)

**Accuracy:** ≤0.5pp gap increase  
**Speedup:** 2.5× on Jetson  
**Memory:** 4× reduction (256→64 MB)

---

## 7. Configuration Files

### 7.1 Training Config (YAML)

```yaml
# config/train.yaml
model:
  encoder_layers: 6
  attention_heads: 8
  embedding_dim: 256
  dropout: 0.1

training:
  batch_size: 128
  learning_rate: 3.0e-4
  warmup_steps: 1000
  max_epochs: 100
  
curriculum:
  p_max: 0.7
  warmup_epochs: 60

data:
  train_instances: 40000
  val_instances: 4000
  problem_sizes: [20, 30, 40, 50]
```

### 7.2 Inference Config (YAML)

```yaml
# config/inference.yaml
hardware:
  device: "mps"  # or "cuda", "cpu"
  mixed_precision: true
  
refinement:
  alpha: 0.7
  candidate_multiplier: 3  # K = 3N
  max_iterations: 2000
  
deployment:
  batch_inference: false
  deterministic: true
```

---

## 8. Implementation Notes

### 8.1 Known Issues

1. **Attention Overflow (N>200):** Use gradient checkpointing
2. **Numerical Instability:** Ensure distance matrix normalization
3. **CUDA OOM (N>150):** Reduce batch size or use CPU offloading

### 8.2 Performance Tips

1. **Precompute Distance Matrix:** Cache d(i,j) before 2-opt
2. **Attention Sparsification:** For N>300, use top-K attention
3. **Dynamic K:** Adjust candidate set size based on time budget

### 8.3 Debugging

**Enable Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Visualize Attention:**
```python
import matplotlib.pyplot as plt
plt.imshow(attention_matrix, cmap='viridis')
plt.colorbar()
plt.title('Attention Weights (Layer 6)')
plt.show()
```

---

## 9. Citation

If you use this architecture, please cite:

```bibtex
@article{cori2026hybrid,
  title={Hybrid Neural–Heuristic TSP Solver with Attention-Guided Refinement for Real-Time AMR Coordination},
  author={BLIND REVIEW},
  journal={Results in Control and Optimization},
  year={2026},
  publisher={Elsevier}
}
```

---

## 10. Contact

**Questions:** BLIND REVIEW  
**Repository:** https://github.com/fnuni/2026rico  
**Paper:** [DOI to be assigned]

---

**Note:** Full implementation code will be released upon paper acceptance, as per our agreement with the industrial partner. The above specifications enable reproduction of the architecture as described in the paper.

**Last Updated:** February 7, 2026  
**Version:** 1.0
