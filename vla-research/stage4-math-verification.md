# Stage 4 — Math Rigor Verification for DCFH

**Date:** 2026-04-29  
**Candidate:** DCFH (Distilled Consistency Flow Head)

---

## Mathematical Claims to Verify

### Claim 1: Consistency distillation preserves the ODE trajectory

**Statement:** If we have a flow matching model that learns velocity field v_θ(x_t, t) defining an ODE dx/dt = v_θ(x, t), and we distill it into a consistency model f_ψ that maps any point on the ODE trajectory to the endpoint x_0, then f_ψ preserves the action distribution.

**Verification:**

The flow matching ODE defines a deterministic mapping from noise ε to action a:
```
a = ε + ∫₀¹ v_θ(x_t, t) dt
```

The consistency function is defined as:
```
f_ψ(x_t, t) = x_τ  for any τ ∈ [0,1] on the same trajectory
```

**Self-consistency loss:**
```
L = E[||f_ψ(x_t, t) - f_ψ(x_{t'}, t')||²]
```
where x_t and x_{t'} are on the same ODE trajectory.

**Problem identified:** The self-consistency loss only enforces that the model maps all points on a trajectory to the same endpoint. It does NOT enforce that the endpoint is correct (i.e., matches the ground truth action). 

**Fix:** Need an additional supervised loss:
```
L_total = L_consistency + λ · ||f_ψ(ε, 0) - a*||²
```
where a* is the ground truth action. This ensures the endpoint matches the expert.

**Verdict:** Claim is valid WITH the supervised loss term. Pure self-consistency distillation is insufficient.

---

### Claim 2: Single-step consistency model matches multi-step flow matching quality

**Statement:** A well-trained consistency model f_ψ(ε, 0) in a single step matches the quality of the flow matching model with 50 steps.

**Verification:**

This is an empirical claim, not a provable theorem. However, we can analyze the error bounds:

**Flow matching error (50 steps):**
```
ε_FM = O(1/N²)  where N = number of steps
```
For N=50: ε_FM ≈ 0.0004 (very small)

**Consistency distillation error:**
```
ε_CD = ε_FM + ε_distill
```
where ε_distill is the distillation error from the self-consistency loss.

**Key insight:** ε_distill depends on:
1. The expressiveness of f_ψ (MLP capacity)
2. The training data coverage (how many ODE trajectories seen)
3. The Lipschitz constant of the true consistency function

**For robot actions:** The action space is typically low-dimensional (7-14 DoF). The ODE trajectories are smooth curves in this space. A 4-layer MLP with skip connections should be sufficiently expressive to learn the consistency function.

**Verdict:** Plausible but not provable a priori. The empirical gap between consistency models and flow matching in image generation is typically 1-3% (Song et al. 2023). For robot actions, the gap may be larger due to tighter geometric constraints.

**Risk:** Mode collapse — if the consistency model learns to map all trajectories to a single "average" action, it fails on multi-modal distributions (e.g., left-grasp vs right-grasp).

**Mitigation:** 
- Use action chunking (predict sequence of actions) to increase the effective dimensionality
- Add diversity regularization to the consistency loss
- Monitor mode coverage during training

---

### Claim 3: The consistency model is a valid single-step generator

**Statement:** f_ψ(ε, 0) directly maps noise to actions without iteration.

**Verification:**

By definition, the consistency function satisfies:
```
f_ψ(x_t, t) = f_ψ(x_{t'}, t')  for all t, t' on the same trajectory
```

At t=0 (the noise endpoint):
```
f_ψ(ε, 0) = f_ψ(a, 1) = a  (the clean action)
```

This is correct by construction. The model at t=0 directly predicts the clean action from noise.

**Verdict:** Valid. This is the definition of a consistency model.

---

### Claim 4: Training stability of consistency distillation for continuous actions

**Statement:** Consistency distillation training is stable for continuous robot action spaces.

**Verification:**

**Potential instabilities:**
1. **Mode collapse:** The model maps all inputs to a single output
2. **Trajectory crossing:** ODE trajectories may cross in high-dimensional spaces (though this is rare for flow matching)
3. **Gradient explosion:** The consistency loss gradient can be large if trajectories are long

**Analysis:**
- Flow matching ODEs are non-crossing by construction (Lipman et al. 2022). This prevents instability from trajectory crossing.
- The self-consistency loss has bounded gradients if the MLP is Lipschitz-constrained.
- Mode collapse is the primary risk (see Claim 2 mitigation).

**Verdict:** Training stability is achievable with:
- Lipschitz constraint on the MLP (spectral normalization)
- Gradient clipping on the consistency loss
- Curriculum: train on easy tasks first (table-top), then hard tasks (long-horizon)

---

## Adversarial Math Review: Counterexamples and Edge Cases

### Counterexample 1: Multi-modal actions
If the expert data has two modes (reach left vs reach right), the flow matching model learns a multi-modal velocity field. The consistency model must preserve both modes.

**Risk:** Consistency distillation may collapse to one mode.
**Test:** Evaluate on LIBERO-Object (which has multi-modal grasps).

### Counterexample 2: Contact-rich manipulation
Contact-rich tasks (e.g., inserting a peg) have highly non-linear dynamics. The ODE trajectories may be complex, making consistency learning harder.

**Risk:** High-frequency contact dynamics may not be captured by a 4-layer MLP.
**Test:** Evaluate on CALVIN (which includes contact tasks).

### Counterexample 3: Out-of-distribution noise
The consistency model is trained on noise from N(0,I). At test time, if the noise distribution shifts slightly, the model may fail.

**Risk:** Distribution shift at test time.
**Mitigation:** Use noise augmentation during training.

---

## Verdict

| Claim | Status | Confidence |
|-------|--------|------------|
| Consistency distillation preserves ODE trajectory | Valid (with supervised loss) | High |
| Single-step matches multi-step quality | Plausible but empirical | Medium |
| Valid single-step generator | Valid by construction | High |
| Training stability | Achievable with mitigations | Medium |

**Overall:** The math is sound but the empirical claims need experimental validation. The primary risk is mode collapse on multi-modal action distributions.

**Recommendation:** Proceed with DCFH, but:
1. Include supervised loss term (not just self-consistency)
2. Add mode-preservation regularization
3. Test on multi-modal benchmarks (LIBERO-Object, CALVIN)
4. Use spectral normalization on the MLP for stability
