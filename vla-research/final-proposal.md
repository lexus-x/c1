# GAUSS-VLA with Conformal Safety Guarantees — Final Proposal

**Date:** 2026-04-29  
**Status:** COMMITTED RESEARCH DIRECTION  
**Target:** IEEE T-RO (primary), IJRR (secondary)  
**Sub-agents:** 4 parallel research agents completed

---

## Executive Summary

After exhaustive research across 4 parallel agents, existing council work (8 killed candidates), and today's full 5-stage pipeline, one direction survives all checks:

**GAUSS-VLA + Conformalized Action Sets on SE(3)**

This fuses the user's existing GAUSS-VLA thesis (Riemannian flow matching + calibrated ensemble uncertainty + adaptive replanning) with a novel conformal prediction layer that provides distribution-free coverage guarantees on SE(3) action sets.

---

## Why This Direction Survives

### What the Council Already Killed (8/8 candidates)
| Candidate | Killed By |
|-----------|-----------|
| P1.1 Consistency-distilled flow | Stage 3 scoop |
| P2.1 Plücker-ray tokens | Stage 3 scoop |
| P3.1 Twin experts + bridge | Stage 3 scoop |
| P4.1 Text-as-action + CFG | Stage 3 scoop |
| P5.1 High-rate re-anchoring | Stage 3 scoop |
| P5.3 Force-token verification | Stage 4.5 (TacVLA, ForceVLA) |
| P2.2 Depth-Forgetting Index | Stage 4.5 (Info-VLA) |
| P4.3 Mixture-of-decoders | Stage 4.5 (DAM-VLA, HiMoE-VLA) |

### What Survives Today's Check
| Direction | Scoop Status | Verdict |
|-----------|-------------|---------|
| DCFH (consistency distillation) | Ada3Drift constrains | ALIVE but weaker |
| PFM-EE (adaptive exit) | GeCO kills it | KILLED |
| IRAN (residual refinement) | ResVLA constrains | ALIVE but repositioned |
| **C1 (conformal action sets)** | **No competitor found** | **ALIVE — strongest** |
| **GAUSS-VLA + Conformal** | **No competitor found** | **ALIVE — recommended** |

---

## The Novelty Triangle

```
Conformal Prediction × VLA × Set-Valued Continuous Actions on SE(3)
```

**This specific combination does not exist in the literature.**

### What Exists (Components in Isolation)
- ✅ Riemannian flow matching on SE(3): RFMP (Jaquier et al., IROS 2024)
- ✅ Conformal prediction on manifolds: Baheri (Feb 2026) — S² only
- ✅ CQR on VLA actions: ReconVLA (Apr 2026) — Euclidean only
- ✅ Ensemble + CP for safety: Tabbara (Nov 2025) — safety filters
- ✅ CP for VLA failure detection: SAFE (NeurIPS 2025)

### What Does NOT Exist (The Fusion)
- ❌ Conformal prediction on SE(3) (non-abelian Lie group, mixed curvature)
- ❌ Geodesic nonconformity scores using bi-invariant metric on SE(3)
- ❌ Conformal guarantees on flow matching outputs (any manifold)
- ❌ Ensemble disagreement as nonconformity score for manifold-valued predictions
- ❌ Lipschitz-based tighter conformal sets on SE(3) via flow matching smoothness

---

## The Three Contributions

### Contribution 1: Riemannian Flow Matching on SE(3) — Theoretical Core

**From GAUSS-VLA (existing thesis):**

Formulate action prediction as a flow on SE(3) × R using exponential coordinates and log map retraction:

```
L_SE(3) = E_{t, X_0, X_1} ||v_θ(X_t, t) - log_{X_t}(X_1)||²_{X_t}
```

**Theorem 1 (Bounded Error of Euclidean Approximation):**
Let f^E be the optimal Euclidean flow matching policy on R⁶ axis-angle and f^* the optimal SE(3) flow matching policy. For any rotation R with ||log(R)|| = θ:

```
E||f^E(R) - f^*(R)|| ≤ C · θ² + O(θ⁴)
```

This predicts empirical gains on rotation-heavy tasks (dial-turn, door-unlock, faucet-open/close, nut-assemble, peg-insert-side, ~15 tasks).

### Contribution 2: Conformalized Action Sets on SE(3) — Novel Theoretical Core

**NEW — does not exist in any prior work:**

Define the nonconformity score on SE(3):

```
s_i = d_SE(3)(Ŷ_i, Y_i) / σ̂(X_i)
```

where d_SE(3) is the geodesic distance under the bi-invariant metric and σ̂(X_i) is a difficulty estimator (from ensemble disagreement in tangent space).

The conformal prediction set:

```
C(x) = { y ∈ SE(3) : d_SE(3)(ŷ(x), y) ≤ σ̂(x) · Q_{1-α}({s_i}) }
```

**Theorem 2 (Conformal Coverage on SE(3)):**
Under exchangeability of {(X_i, Y_i)} with Y_i ∈ SE(3):

```
P(Y_{n+1} ∈ C(x_{n+1})) ≥ 1 - α
```

**Theorem 3 (Equivariance Under Rigid Transformations):**
If the data distribution is equivariant under group G acting on SE(3), the conformal prediction sets are G-equivariant:

```
C(g · x) = g · C(x)  for all g ∈ G
```

**Theorem 4 (Tightness Bound via Lipschitz):**
If the flow matching velocity field v_t has Lipschitz constant L on SE(3), the expected volume of C(x) is:

```
Vol(C(x)) = O(L · σ̂(x) / √M)
```

where M is ensemble size. **Smaller models with lower L produce tighter sets.**

### Contribution 3: Uncertainty-Driven Adaptive Replanning — Practical Core

**From GAUSS-VLA (existing thesis):**

Replace ACT's fixed H=8 chunking with adaptive horizon:

```
H_t = H_max · σ(u_t)  where u_t = Var_{i=1}^3[f_{θ_i}(s_t, ℓ)]
```

When conformal set size is large (high uncertainty) → shorten horizon, replan sooner.
When conformal set size is small (low confidence) → extend horizon, commit to actions.

---

## Architecture

```
[Vision Encoder: SigLIP-400M (frozen)] ─┐
[Language Encoder: T5/frozen]            ├→ [Cross-Attention Mixer 5M]
[Proprioception Tokenizer 2M]           ┘         ↓
                                    [SE(3) Flow Matching Head 35M] × 3 (ensemble)
                                              ↓
                                    [Conformal Calibration Module 0M (post-hoc)]
                                              ↓
                                    Output: Action Set C(x) ⊂ SE(3) with coverage ≥ 1-α
```

### Parameter Budget

| Component | Params | Trainable |
|-----------|--------|-----------|
| SigLIP vision encoder | 400M | No (frozen) |
| Cross-attention mixer | 5M | Yes |
| SE(3) flow matching head × 3 | 105M (35M each) | Yes |
| Conformal calibration | 0M | Post-hoc |
| **Total** | **~510M** | **~110M** |

⚠️ **Over budget by ~10M.** Mitigation: Use SigLIP-256M (smaller variant) or share vision encoder across ensemble heads. With sharing: **~405M total.**

---

## Expected Performance

### Head-to-Head vs Baselines

| Metric | SmolVLA (450M) | OpenVLA (7B) | GAUSS-VLA (405M) | Δ vs SmolVLA |
|--------|---------------|-------------|-----------------|-------------|
| LIBERO Avg | ~95% | ~97% | ~95-97% | 0 to +2% |
| CALVIN ABC | ~4.0 | ~4.3 | ~4.1-4.3 | +0.1 to +0.3 |
| Rotation-heavy tasks | ~70% (est.) | ~75% (est.) | ~80-85% (est.) | **+10-15%** |
| Coverage (1-α=0.9) | N/A | N/A | ≥90% (guaranteed) | **New metric** |
| Avg set size | N/A | N/A | ~2-5 actions | **New metric** |
| Failure detection (AUC) | ~0.7 (SAFE) | ~0.75 (SAFE) | ~0.85-0.9 | **+15-20%** |

### New Metrics Introduced
1. **Action Coverage**: P(a* ∈ C(x)) — must be ≥ 1-α
2. **Average Set Size**: Smaller = more confident
3. **Geodesic Set Diameter**: Tightness of the prediction set on SE(3)
4. **action-ECE**: Calibration error of the ensemble uncertainty

---

## Compute Budget

| Phase | A100-Days | Notes |
|-------|-----------|-------|
| Pre-training (frozen backbone) | 0 | Using existing SigLIP/T5 |
| SE(3) flow matching training | 5-7 | 3 heads × 35M params |
| Conformal calibration | 1-2 | Post-hoc, no retraining |
| Evaluation (LIBERO + CALVIN + real) | 3-5 | 3 seeds × multiple benchmarks |
| Real robot experiments | 2-3 | ~200 demonstrations |
| **Total** | **11-17** | ⚠️ Tight but feasible |

---

## Scoop Risk Assessment

| Competitor | Risk | Reasoning |
|-----------|------|-----------|
| ReconVLA (Apr 2026) | 🔴 HIGH | 12 days old. Uses CQR on VLA actions in Euclidean space. Adding geodesic distances is a natural v2. |
| Baheri (Feb 2026) | 🟡 MEDIUM | Conformal on manifolds (S²). Could extend to Lie groups but hasn't. |
| RFMP group (Jaquier) | 🟡 MEDIUM | Knows SE(3) flow matching. Adding CP is natural but no interest shown. |
| General CP + robotics | 🟡 MEDIUM | Field is hot (SAFE, ReconVLA, Tabbara in 2025-2026). |

**Overall scoop risk: MEDIUM.** Window: 6-12 months. Must submit within 6 months.

---

## Adversarial Review: Why Would T-RO Reject This?

### Reviewer 2 Attacks:

1. **"Incremental over Baheri"** — Conformal on manifolds exists (S²). SE(3) is a straightforward generalization.
   - **Defense:** SE(3) is NOT a simple generalization. It's a semidirect product SO(3) ⋉ ℝ³ with mixed sectional curvature and non-commutative group structure. The geodesic computation, bi-invariant metric choice, and volume corrections are non-trivial. Reviewer must show the generalization is trivial — we argue it isn't.

2. **"ReconVLA already does this"** — ReconVLA applies CP to VLA actions.
   - **Defense:** ReconVLA uses Euclidean CQR on action tokens. We use geodesic CQR on SE(3). The geometry matters: Euclidean CP on rotations penalizes equivalent representations (q and -q are the same rotation). Our CP respects the manifold structure.

3. **"Set-valued actions are impractical"** — What does the robot DO with a set?
   - **Defense:** The planner selects the action closest to the set center that satisfies joint limits and collision constraints. We demonstrate this with a robust MPC planner. Set size provides real-time uncertainty feedback for adaptive replanning.

4. **"The Lipschitz argument is hand-wavy"** — Smaller models don't necessarily have lower Lipschitz constants.
   - **Defense:** We empirically measure the spectral norm of weight matrices during training. We show the correlation between model size and Lipschitz constant on the specific architecture. If the monotonic relationship doesn't hold, we drop the claim and rely on empirical tightness.

---

## Alternative/Backup: VLA Zoo Benchmark Paper

If GAUSS-VLA + Conformal faces insurmountable obstacles (e.g., ReconVLA v2 scoops the conformal component), pivot to:

**"VLA Zoo: A Systematic Evaluation of Vision-Language-Action Architectures"**

- Evaluate 8-12 VLA variants head-to-head on same tasks, same hardware, same protocol
- Genuine gap: no paper does this (Wang et al. survey covers data, not models)
- Compute-feasible: use public checkpoints + fine-tuning
- High community value, strong T-RO fit
- Can be done in parallel as a fallback

---

## Implementation Roadmap

### Month 1: Foundation
- [ ] Implement SE(3) flow matching head on existing SmolVLA/OpenVLA backbone
- [ ] Train 3-head ensemble on LIBERO + CALVIN
- [ ] Validate Theorem 1 (Euclidean vs SE(3) error bound)

### Month 2: Conformal Layer
- [ ] Implement geodesic nonconformity score
- [ ] Calibrate on held-out data
- [ ] Validate Theorem 2 (coverage guarantee)
- [ ] Measure set sizes across tasks

### Month 3: Planner + Evaluation
- [ ] Implement robust MPC planner that consumes conformal sets
- [ ] Full benchmark evaluation (LIBERO, CALVIN, rotation-heavy subset)
- [ ] Real robot experiments (~200 demos)
- [ ] Validate Theorem 4 (tightness bound)

### Month 4: Paper Writing
- [ ] Write §I-III (GAUSS-VLA core)
- [ ] Write §IV (Conformalized Action Sets on SE(3))
- [ ] Write §V (Planner integration)
- [ ] Write §VI (Experiments)
- [ ] Internal review + revision

### Month 5: Submission
- [ ] Final revision
- [ ] Submit to IEEE T-RO
- Target: September 2026

---

## Files in This Repository

- `stage1-unsolved-problems.md` — VLA landscape analysis
- `stage2-candidates.md` — Three candidate architectures (DCFH, IRAN, PFM-EE)
- `stage3-scoop-check.md` — Scoop check on original candidates
- `stage3-conformal-scoop-check.md` — Scoop check on conformal idea
- `stage4-math-verification.md` — Math verification for DCFH
- `stage5-adversarial-debate.md` — Adversarial debate + final decision
- **`final-proposal.md`** — This document (the committed direction)

---

## References

### Core VLA Papers
- SmolVLA (Hugging Face, 2025) — 450M params, flow matching
- OpenVLA (Kim et al., 2024) — 7B params, autoregressive
- π0 (Black et al., 2026) — flow matching VLA
- RDT-1B (2024) — diffusion foundation model

### Riemannian Flow Matching
- Flow Matching on General Geometries (Chen et al., NeurIPS 2024)
- RFMP (Jaquier et al., IROS 2024) — SE(3) flow matching for robot policies

### Conformal Prediction
- Baheri & Shahbazi (Feb 2026) — Conformal on manifolds (S²)
- SAFE (NeurIPS 2025) — CP for VLA failure detection
- ReconVLA (Apr 2026) — CQR on VLA action tokens (Euclidean)
- ConformalDAgger (ICLR 2025 sub) — CP intervals for IL

### Safety
- VLSA/AEGIS (Dec 2025) — CBF safety filters for VLA
- VLA Safety Survey (Li et al., Apr 2026) — lists "certified robustness" as open problem

### Killed Candidates (Council)
- PRISM-VLA, STRATOS-VLA, PROGRESS-VLA — killed in prior sessions
- 8 Stage 2-4.5 candidates — killed by scoop check
