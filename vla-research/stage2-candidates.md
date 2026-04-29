# Stage 2 — Three Candidate Architectures for Fast Action Decoding

**Date:** 2026-04-29  
**Problem:** Action Representation Bottleneck — no sub-500M VLA has a fast, high-fidelity action decoder  
**Target:** 1-3 step decoder matching 50-step diffusion quality at ≥5× inference speedup

---

## Candidate A: Distilled Consistency Flow Head (DCFH)

**Thesis:** Distill a pre-trained flow matching action decoder into a consistency model that maps noise → action in 1 step, using the frozen VLM backbone as conditioning.

### Architecture

```
[VLM Backbone 250M (frozen)] → visual + language tokens
        ↓
[Cross-Attention Mixer 5M] → fused latent z
        ↓
[Consistency Action Head 30M] 
  - Input: z + noise ε ~ N(0,I)
  - Architecture: 4-layer MLP with skip connections
  - Output: clean action a ∈ R^d (single step)
```

### Training

- Stage 1: Train normal flow matching head (50 steps) with frozen backbone
- Stage 2: Consistency distillation — enforce self-consistency:
  - f_θ(z, ε) = f_θ(z, ε') for nearby ε, ε' along ODE trajectory
  - Loss: L_consistency + L_trajectory_match + L_smoothness

### Parameter Count

| Component | Params | Trainable |
|-----------|--------|-----------|
| VLM Backbone | 250M | No (frozen) |
| Cross-Attention Mixer | 5M | Yes |
| Consistency Action Head | 30M | Yes |
| **Total** | **285M** | **35M** |

### Inference

1 forward pass through VLM + 1 pass through action head = **~2 steps total**  
Latency: ~5ms per action (vs ~50ms for 50-step flow matching)

### Expected Performance

| Metric | SmolVLA (450M, 50-step) | DCFH (285M, 1-step) | Δ |
|--------|------------------------|---------------------|---|
| LIBERO Avg | ~95% | ~92-94% | -1-3% |
| CALVIN ABC | ~4.0 | ~3.7-3.9 | -0.1-0.3 |
| Inference latency | ~50ms | ~5ms | **10× faster** |
| Params | 450M | 285M | **37% smaller** |

### Training Cost

~3-5 days on 8×A100 (frozen backbone, only distilling 35M params)

### Risk Assessment

- **High risk**: Consistency distillation in continuous action space is underexplored
- In image generation, consistency distillation works because the data manifold is well-understood
- Robot actions have tighter geometric constraints — the consistency loss might not preserve manipulation precision
- **Mitigation**: Add action-space smoothness regularization; evaluate on both LIBERO (table-top) and CALVIN (long-horizon)

### Novelty Check

- Consistency distillation applied to images (Song et al. 2023) and video — **never to VLA action decoders**
- The combination: flow-matching distillation + frozen VLM conditioning + continuous robot actions = **untested**
- **Scoop risk: LOW** — no prior work at this intersection

---

## Candidate B: Iterative Residual Action Network (IRAN) ⭐ RECOMMENDED

**Thesis:** Train a lightweight iterative refinement network that predicts residual corrections. Each step refines the previous prediction. 2-3 steps converges to high-quality actions.

### Architecture

```
[VLM Backbone 250M (frozen)] → visual + language tokens
        ↓
[Cross-Attention Mixer 5M] → fused latent z
        ↓
[Residual Action Network 40M]
  Step 0: a_0 = MLP_init(z)  — coarse initial action
  Step 1: δ_1 = RefineNet(z, a_0, t_1) → a_1 = a_0 + δ_1
  Step 2: δ_2 = RefineNet(z, a_1, t_2) → a_2 = a_1 + δ_2
  Step 3: δ_3 = RefineNet(z, a_2, t_3) → a_3 = a_2 + δ_3
  
  RefineNet: 
    - 2-layer MLP with FiLM conditioning on timestep t
    - Predicts residual in action space
    - Shares weights across steps (recurrent refinement)
```

### Training

- Supervised on expert trajectories
- Loss: Σ_k w_k · ||a_k - a*||²  (weighted sum across refinement steps)
- Curriculum: start with 1-step, increase to 3-step during training
- Auxiliary: trajectory smoothness loss on final prediction

### Parameter Count

| Component | Params | Trainable |
|-----------|--------|-----------|
| VLM Backbone | 250M | No (frozen) |
| Cross-Attention Mixer | 5M | Yes |
| Refinement Network | 40M | Yes |
| **Total** | **295M** | **45M** |

### Inference

1 VLM pass + 2-3 refinement passes = **3-4 steps total**  
Latency: ~8-12ms per action

### Expected Performance

| Metric | SmolVLA (450M, 50-step) | IRAN (295M, 3-step) | Δ |
|--------|------------------------|---------------------|---|
| LIBERO Avg | ~95% | ~93-95% | 0 to -2% |
| CALVIN ABC | ~4.0 | ~3.8-4.0 | 0 to -0.2 |
| Inference latency | ~50ms | ~10ms | **5× faster** |
| Params | 450M | 295M | **34% smaller** |

### Training Cost

~5-7 days on 8×A100

### Theoretical Contribution

**Convergence Theorem (sketch):**  
Under Lipschitz continuity of the refinement mapping, 3-step residual refinement converges to within ε of the fixed point if the spectral radius of the Jacobian ∂δ/∂a < 1.

**Proof direction:**
- Define the refinement operator T: a → a + δ(z, a, t)
- If ||JT||_2 = ρ < 1 (contraction), then by Banach fixed-point theorem, T has a unique fixed point a*
- After k steps: ||a_k - a*|| ≤ ρ^k / (1-ρ) · ||a_1 - a_0||
- For ρ = 0.5 and k=3: error ≤ 0.125 · ||a_1 - a_0|| (converges fast)
- The FiLM conditioning on t controls ρ — the model learns to be more conservative at later steps

**Implication:** The theorem gives a principled way to choose the number of refinement steps based on the desired precision-action tradeoff.

### Risk Assessment

- **Medium risk**: Refinement might collapse if initial prediction is very wrong
- Residual corrections may not recover from bad initialization
- **Mitigation**: 
  - Train with noise injection on a_0 to improve robustness
  - Use curriculum: 1-step first (good initialization), then 2-step, then 3-step
  - Monitor convergence rate ρ during training; if ρ→1, the model is not learning to refine

### Novelty Check

- Iterative refinement exists in object detection (Cascade R-CNN) and image restoration
- **Never applied to VLA action decoding**
- Key novelty: weight-shared residual refinement conditioned on VLM features for continuous robot actions
- Distinct from diffusion: diffusion iterates on noise, IRAN iterates on actions directly
- **Scoop risk: MEDIUM** — the paradigm is known, but the application to VLA is new

---

## Candidate C: Predictive Flow Matching with Early Exit (PFM-EE)

**Thesis:** Train a flow matching decoder normally, but add learned early-exit heads at intermediate denoising steps. A lightweight gating network decides when the action is "good enough" and exits early.

### Architecture

```
[VLM Backbone 250M (frozen)] → visual + language tokens
        ↓
[Cross-Attention Mixer 5M] → fused latent z
        ↓
[Flow Matching Decoder 35M] (shared weights)
  - Standard flow matching: dx/dt = v_θ(x_t, t, z)
  - With K early-exit heads at t ∈ {0.1, 0.2, 0.3, ...}
  
[Exit Gate Network 5M]
  - Input: current action prediction a_t, latent z, step t
  - Output: confidence score c ∈ [0,1]
  - If c > threshold τ: exit early, return a_t
  - Otherwise: continue denoising
```

### Training

- Stage 1: Train full flow matching decoder (50 steps) normally
- Stage 2: Add exit heads, train with:
  - L_flow: standard flow matching loss at each step
  - L_exit: binary cross-entropy for exit decisions
  - L_quality: MSE on early-exit predictions vs final prediction
- Stage 3: Fine-tune exit gate with RL reward = success_rate - λ·latency

### Parameter Count

| Component | Params | Trainable |
|-----------|--------|-----------|
| VLM Backbone | 250M | No (frozen) |
| Cross-Attention Mixer | 5M | Yes |
| Flow Matching Decoder | 35M | Yes |
| Exit Gate Network | 5M | Yes |
| **Total** | **295M** | **45M** |

### Inference

Adaptive — **2-5 steps** for easy tasks, up to 20 for hard tasks. Average ~3-5 steps.  
Latency: ~8-15ms average

### Expected Performance

| Metric | SmolVLA (450M, 50-step) | PFM-EE (295M, adaptive) | Δ |
|--------|------------------------|--------------------------|---|
| LIBERO Avg | ~95% | ~94-96% | -1% to +1% |
| CALVIN ABC | ~4.0 | ~3.9-4.1 | -0.1 to +0.1 |
| Inference latency | ~50ms (fixed) | ~10ms avg (adaptive) | **5× faster avg** |
| Hard task latency | ~50ms | ~30ms (exits later) | ~1.7× faster |
| Params | 450M | 295M | **34% smaller** |

### Training Cost

~7-10 days on 8×A100 (two-stage training)

### Theoretical Contribution

**Optimal Exit Policy:**  
For a task distribution with entropy H(tasks), the optimal exit policy minimizes E[latency] subject to E[success_rate] ≥ 1-δ.

This reduces to a threshold policy on the exit confidence score:
- τ* = f(task_entropy, model_calibration_error)
- Well-calibrated models can use lower thresholds safely
- The exit policy is a form of anytime algorithm — quality improves monotonically with compute

### Risk Assessment

- **Medium-High risk**: Exit gate is the weakest link
- If it exits too early on hard tasks → performance degrades
- If it exits too late → no speedup
- RL reward shaping is tricky
- **Mitigation**:
  - Use supervised exit labels from expert trajectories (not just RL)
  - Start with conservative thresholds, anneal during training
  - Add a "minimum steps" floor (never exit before step 2)

### Novelty Check

- Early exit applied to LLMs and classifiers — **never to flow matching / diffusion action decoders**
- Adaptive computation allocation for robot actions is novel
- Combination of flow matching + learned exit gating + RL-tuned threshold is novel
- **Scoop risk: LOW** — adaptive flow matching for actions is unexplored

---

## Comparison Matrix

| | DCFH (Consistency) | IRAN (Residual) ⭐ | PFM-EE (Early Exit) |
|---|---|---|---|
| **Core idea** | Distill flow → 1 step | Refine residuals iteratively | Adaptive step count |
| **Steps** | 1 | 3-4 | 2-5 (adaptive) |
| **Speedup** | 10× | 5× | 5× avg |
| **Quality risk** | High (distillation loss) | Medium (convergence) | Low (preserves full model) |
| **Training cost** | Low (3-5 A100-days) | Medium (5-7 days) | High (7-10 days) |
| **Novelty** | High (first consistency VLA) | Medium (known paradigm, new domain) | High (adaptive flow matching) |
| **Theoretical contribution** | Weak (empirical) | Strong (convergence proof) | Medium (optimal exit policy) |
| **Scoop risk** | Low | Medium | Low |

## Decision

**Primary: IRAN (Candidate B)** — most likely to produce a working system with clear contribution  
**Secondary: PFM-EE (Candidate C)** — highest upside if exit gate works  
**Backup: DCFH (Candidate A)** — cleanest story if distillation works

Next: Stage 3 — Scoop check on IRAN
