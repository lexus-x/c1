# Stage 5 — Adversarial Debate + Final Decision

**Date:** 2026-04-29  
**Surviving candidates:** DCFH, C1, IRAN (repositioned)  
**Killed:** PFM-EE (killed by GeCO)

---

## Adversarial Attack on DCFH

### Attack 1: "Reviewer-2 Perspective — Why Would I Reject This?"

**Reviewer 2 says:**

"This paper proposes consistency distillation for VLA action decoders. My concerns:

1. **Mode collapse is not adequately addressed.** The paper claims consistency distillation preserves multi-modal action distributions, but the only mitigation is 'diversity regularization.' In image generation, consistency models are known to produce blurrier outputs than their teacher models. In robot manipulation, 'blurry' means 'averaged trajectory into an obstacle.' The experiments on LIBERO (table-top, mostly uni-modal) do not stress-test this sufficiently. What about bimanual tasks with genuinely multi-modal solutions?

2. **The 10× speedup claim is misleading.** The 50ms baseline assumes a naive 50-step flow matching implementation. With existing acceleration techniques (DySL-VLA achieves 3.75× speedup, GeCO achieves adaptive steps), the practical speedup of a single-step model is more like 3-5×, not 10×. The paper should compare against optimized baselines, not worst-case implementations.

3. **The theoretical contribution is weak.** The paper acknowledges that consistency distillation is empirical, not provable. The convergence claim relies on 'the MLP being expressive enough' — this is not a theorem, it's a hope. For a Q1 venue, I expect either strong theory or strong empirical results. This paper has neither in abundance.

4. **Why not just use Ada3Drift?** Ada3Drift (arxiv 2603.11984) achieves single-step generation with training-time drifting, which avoids the distillation error entirely. The paper should explain why consistency distillation is preferable to training-time refinement."

**Defense:**
1. Mode collapse: Test on CALVIN (bimanual) and LIBERO-Object (multi-modal grasps). Add explicit mode-coverage metrics.
2. Speedup: Compare against DySL-VLA and GeCO, not just naive baselines. Honest speedup: 3-5×.
3. Theory: Drop the "provable" framing. Position as empirical contribution with theoretical motivation.
4. Ada3Drift: Different domain (3D point-cloud, not VLA), different technique (training-time drifting, not distillation). Consistency distillation is more general (works with any pre-trained flow model).

---

### Attack 2: "Scoop-Risk — What Obsoletes This in 6 Months?"

**Most likely obsoleting paper:**

"ConsistencyVLA: Consistency Distillation for Vision-Language-Action Models" — if someone publishes this exact idea before you. The window is 3-6 months.

More realistically: A large lab (Physical Intelligence, Hugging Face) releases a single-step VLA as part of a larger system paper. They don't focus on the action decoder specifically, but their solution implicitly achieves the same result. This is the "steamroller" scenario — you get scooped not by a direct competitor, but by a system paper that happens to include your contribution as a component.

**Mitigation:** Speed. Submit within 2 months. The consistency-distillation-for-VLA window is open NOW but closing.

---

## Adversarial Attack on C1

### Attack 1: "Reviewer-2 Perspective"

**Reviewer 2 says:**

"This paper applies conformal prediction to VLA action spaces. My concerns:

1. **What does the robot DO with the action set?** The paper proposes set-valued action predictions but the practical utility is unclear. If the robot picks the most likely action and uses set size as uncertainty, this is just SAFE with extra computation. If the robot uses a fallback strategy, the paper needs to specify and evaluate that strategy. Without a concrete use case, this is a solution looking for a problem.

2. **Continuous action spaces break the CP paradigm.** Conformal prediction was designed for classification (finite label sets) and regression (scalar outputs). In 7-14 DoF continuous action spaces, the 'set' is a region in R^d. How do you represent this region? As a bounding box? A level set of the nonconformity score? The paper needs to specify the set geometry and show it's computationally tractable.

3. **The Lipschitz argument is hand-wavy.** The claim that 'smaller models have lower Lipschitz constants and therefore tighter conformal sets' needs proof. In practice, smaller models may have HIGHER Lipschitz constants in certain layers due to less overparameterization. The relationship between model size and Lipschitz constant is not monotonic.

4. **Evaluation is unclear.** How do you show set-valued predictions are better than single predictions + SAFE? The metrics (coverage, set size) are internal to the method. The paper needs external metrics: task success rate, failure rate, safety incidents."

**Defense:**
1. Action set utility: Design a concrete fallback strategy (e.g., if set size > threshold, execute conservative pre-computed action). Evaluate on safety metrics.
2. Continuous CP: Use conformalized quantile regression (CQR) for multi-dimensional outputs. This is established in the CP literature.
3. Lipschitz: Formalize the argument with actual Lipschitz constant estimation (spectral norm of weight matrices). If the monotonic relationship doesn't hold, drop the claim.
4. Evaluation: Compare against SAFE + single-action baselines on failure rate and task success.

### Attack 2: "Scoop-Risk"

**Most likely obsoleting paper:**

SAFE (NeurIPS 2025) already uses CP for VLA failure detection. If the SAFE authors extend their work to set-valued predictions (natural next step), C1 is scooped. Also, CASP (action sets in RL) could be extended to VLA.

**Mitigation:** The SAFE authors focus on failure detection, not action representation. But the extension is natural. Speed matters.

---

## Adversarial Attack on IRAN (Repositioned)

### Attack 1: "Reviewer-2 Perspective"

**Reviewer 2 says:**

"ResVLA (arxiv 2604.21391) already proposed residual refinement for VLA. This paper claims to make it 'faster' by using weight-shared MLPs instead of diffusion bridges. But:

1. ResVLA's diffusion bridge captures the stochastic nature of high-frequency residuals. A deterministic MLP cannot model this stochasticity. The paper trades fidelity for speed — is this actually better than ResVLA with fewer diffusion steps?

2. The convergence theorem assumes the refinement operator is a contraction. But in practice, the FiLM conditioning may not ensure contraction. The theorem is vacuous if the assumption doesn't hold.

3. The positioning as 'ResVLA for edge deployment' is a weak contribution. It's an engineering optimization, not a research contribution."

**Defense:**
1. Stochasticity: Add a small noise injection to the MLP to capture residual stochasticity. Or use a variational formulation.
2. Convergence: Empirically measure the spectral radius during training. If it's <1, the theorem applies.
3. Contribution: The convergence theorem + empirical speedup + sub-500M evaluation is more than engineering.

**Verdict: IRAN survives but is weaker than DCFH or C1.**

---

## Final Comparison

| Criterion | DCFH | C1 | IRAN |
|-----------|------|-----|------|
| Novelty | High (first consistency VLA) | High (first set-valued VLA) | Medium (ResVLA derivative) |
| Scoop risk | Low | Low-Medium | Medium (ResVLA) |
| Speedup | 10× (3-5× honest) | N/A (safety, not speed) | 5× |
| Theory | Weak (empirical) | Strong (CP guarantees) | Medium (convergence) |
| Practical impact | High (faster inference) | High (safer deployment) | Medium |
| Compute cost | 3-5 A100-days | 3 A100-days | 5-7 A100-days |
| Evaluation clarity | High (speed + success rate) | Medium (needs fallback design) | High |
| Q1 potential | Strong | Strong (if fallback designed) | Medium |

---

## Final Decision

### Primary: **DCFH** — Distilled Consistency Flow Head

**Why:**
1. Cleanest novelty window — no one has done consistency distillation for VLA
2. Strongest practical impact — 3-5× speedup is directly measurable
3. Lowest scoop risk — niche enough that big labs won't prioritize it
4. Simplest to implement — freeze backbone, distill action head
5. Clear evaluation — speed + success rate on standard benchmarks

**What makes it Q1-worthy:**
- First single-step VLA action decoder via consistency distillation
- Honest comparison against optimized baselines (not strawmen)
- Real robot demo showing real-time control
- The "sub-500M advantage" argument (smaller action head = easier to distill)

**Key risk to manage:** Mode collapse on multi-modal tasks. Mitigate with mode-coverage metrics and multi-modal benchmark evaluation.

### Secondary: **C1** — Conformal Action Sets

**Why it's backup:**
- Novel formalism with strong theoretical appeal
- But: needs concrete fallback strategy design
- But: continuous action set representation is non-trivial
- But: may be perceived as "just CP applied to VLA"

**When to use C1:** If DCFH experiments show mode collapse issues, pivot to C1 as a safety-focused contribution. Or: combine C1 + DCFH (fast decoder + conformal safety wrapper).

### Killed: PFM-EE, IRAN (as primary)

---

## Recommended Thesis Statement

**"Distilled Consistency Flow Head (DCFH): Real-Time Sub-500M Vision-Language-Action via Single-Step Action Decoding"**

We propose DCFH, the first consistency-distilled action decoder for Vision-Language-Action models. By distilling a pre-trained flow matching action head into a single-step consistency model, DCFH achieves 3-5× inference speedup over standard flow matching while maintaining competitive task success rates on LIBERO and CALVIN benchmarks. The full model operates at 285M parameters with 35M trainable, enabling real-time robotic control on consumer-grade GPUs. We provide theoretical motivation via Lipschitz analysis of the consistency function and empirical validation on both simulation and real-world manipulation tasks.
