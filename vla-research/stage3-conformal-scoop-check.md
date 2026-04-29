# Stage 3 — Scoop Check: Conformal Action Sets for VLA (C1)

**Date:** 2026-04-29  
**Candidate:** C1 — Conformal Action Sets for VLA Models

---

## Near-Hit #1: SAFE — Multitask Failure Detection for VLA Models

**Paper:** NeurIPS 2025  
**Source:** vla-safe.github.io

### What it does
- Failure detector for VLA models (OpenVLA, π0, π0-FAST)
- Learns from VLA internal features → predicts failure score
- Uses functional conformal prediction for time-varying thresholds
- Tested on LIBERO, SimplerEnv, real-world Franka

### Does it kill C1?

**NO — fundamentally different problem.**

| Aspect | SAFE | C1 |
|--------|------|-----|
| Goal | Detect WHEN the policy fails | Output WHAT actions are valid |
| Output | Binary fail/success signal | Set of candidate actions |
| CP usage | Threshold calibration for failure score | Coverage guarantee for action sets |
| When used | Post-hoc monitoring (after action) | During action selection (before action) |
| Modifies action? | No (policy runs normally, just monitored) | Yes (returns action set, not single action) |

**Assessment:** SAFE uses CP as a calibration tool for failure detection thresholds. C1 uses CP to construct set-valued predictions in action space. These are orthogonal contributions. SAFE is a safety monitor; C1 is a new action representation.

---

## Near-Hit #2: FAIL-Detect — Uncertainty-Aware Runtime Failure Detection

**Paper:** RSS 2025 (arxiv 2503.08558)

### What it does
- Two-stage failure detection for imitation learning
- Distills policy inputs/outputs into scalar signals
- Uses CP for uncertainty quantification with statistical guarantees

### Does it kill C1?

**NO — same distinction as SAFE.**

FAIL-Detect detects failures post-hoc. C1 changes the action representation to set-valued predictions. FAIL-Detect answers "did I fail?"; C1 answers "what are my valid options?"

---

## Near-Hit #3: CASP — Coupled Action-Set Pessimism

**Paper:** arxiv (recent, exact ID unclear)

### What it does
- Support-aware offline RL
- Uses conformal risk control on action subsets
- Calibrated prediction sets for offline RL

### Does it kill C1?

**PARTIALLY — constrains the novelty.**

CASP uses action sets with CP, but in offline RL, not VLA. Key differences:

| Aspect | CASP | C1 |
|--------|------|-----|
| Domain | Offline RL | VLA (imitation learning + VLM backbone) |
| Action set purpose | Pessimistic policy optimization | Coverage guarantee for safe deployment |
| CP integration | During training (offline RL optimization) | During inference (post-hoc calibration) |
| Model-agnostic? | No (tied to offline RL framework) | Yes (wraps any VLA) |

**Assessment:** CASP shows action sets + CP exist in RL, but the VLA context, model-agnostic wrapper approach, and sub-500M focus are novel.

---

## Near-Hit #4: Conformal Prediction in Autonomous Driving

**Status:** Extensive use of CP for trajectory prediction and uncertainty in self-driving (multiple papers). But NOT applied to robot manipulation action policies.

### Does it kill C1?

**NO — different domain, different action space.**

---

## Novelty Assessment

### What IS novel about C1:
1. **First set-valued action prediction for VLA models** — nobody has done this
2. **Model-agnostic wrapper** — works with SmolVLA, OpenVLA, π0, any VLA
3. **Sub-500M theoretical advantage** — tighter conformal sets for smaller models (Lipschitz argument)
4. **New evaluation metrics** — action coverage, set size, failure detectability

### What is NOT novel:
1. Conformal prediction itself (well-established)
2. CP for robot failure detection (SAFE, FAIL-Detect)
3. Action sets in RL (CASP)
4. Uncertainty quantification for policies (many papers)

### Novelty verdict: **HIGH** — the specific combination (CP + set-valued actions + VLA + sub-500M advantage) is unclaimed.

---

## Scoop Risk Assessment

| Threat | Risk Level | Reasoning |
|--------|-----------|-----------|
| SAFE/FAIL-Detect expanding to action sets | LOW | Their focus is failure detection, not action representation |
| CASP expanding to VLA | LOW | CASP is tied to offline RL, VLA is imitation learning |
| New paper in next 6 months | MEDIUM | CP is trendy; someone might try this |
| Autonomous driving CP → robotics | LOW | Different action spaces and safety requirements |

**Overall scoop risk: LOW-MEDIUM**

---

## Strengths for Q1

1. **Clear formalism**: CP is mathematically rigorous, distribution-free
2. **Practical value**: Robot safety is a hot topic; set-valued predictions enable fallback strategies
3. **Novel theoretical insight**: Smaller models → tighter conformal sets (Lipschitz argument)
4. **Easy to evaluate**: New metrics (coverage, set size) + standard benchmarks (LIBERO, CALVIN)
5. **Low compute cost**: ~3 A100-days (post-hoc calibration, no retraining)
6. **Complementary to other work**: Can be combined with fast action decoders (DCFH/IRAN)

## Weaknesses for Q1

1. **Action set size in continuous spaces**: In 7-14 DoF continuous action spaces, the "set" is a region in R^d. How do you represent and sample from this set efficiently?
2. **Practical utility**: Does a robot actually USE a set of actions? Or just pick the most likely one? If it picks the most likely, then the set is just an uncertainty indicator (similar to SAFE).
3. **May be perceived as incremental**: CP is well-known; applying it to VLA might be seen as "just another application" rather than a fundamental contribution.
4. **Evaluation challenge**: How do you show that set-valued predictions are BETTER than single predictions + failure detection?

---

## Recommendation

**C1 is viable but has risks.** The novelty window is real, but:

1. **The "action set" concept needs a concrete use case** — what does the robot DO with the set? If the answer is "pick the best one and use the set size as uncertainty," then it's essentially SAFE with extra steps.

2. **The strongest version of C1** would be: "The robot uses the action set to implement a fallback strategy — if the primary action is risky (large set), it executes a conservative alternative." This is a genuine safety contribution.

3. **The weakest version** is: "We compute conformal sets and report coverage metrics." This is a benchmarking contribution, not a system contribution.

**For Q1, C1 needs:**
- A concrete fallback strategy that uses the action set
- Real robot experiments showing fewer failures
- The Lipschitz-sub-500M argument formalized as a theorem
- Comparison against SAFE + standard baselines
