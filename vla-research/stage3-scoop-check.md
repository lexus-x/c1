# Stage 3 — Scoop Check on IRAN

**Date:** 2026-04-29  
**Candidate under check:** IRAN (Iterative Residual Action Network)

---

## Near-Hit #1: ResVLA — "From Noise to Intent: Anchoring Generative VLA Policies with Residual Bridges"

**Paper:** arxiv 2604.21391 (published 6 days ago, April 23, 2026)  
**Authors:** Yaoyu He, Zemin Yang, Pengfei Tian, Yifan Huang, Qingqiu Huang, Xinge Zhu, Yuexin Ma

### What it does
- Decomposes robotic control into low-frequency intent + high-frequency residual via spectral analysis
- Predicts deterministic low-frequency anchor from VLM features
- Refines only high-frequency residual using a diffusion bridge
- Paradigm shift: "Generation-from-Noise" → "Refinement-from-Intent"

### Does it kill IRAN?

**Partially. It constrains the novelty but does NOT kill it.**

Key differences:
| Aspect | ResVLA | IRAN (our proposal) |
|--------|--------|---------------------|
| Decomposition | Spectral (FFT-based) frequency decoupling | Learned coarse-to-fine via MLP_init |
| Refinement method | Diffusion bridge (multiple denoising steps) | Weight-shared residual MLP (2-3 steps) |
| Inference speed | Still multi-step (diffusion bridge) | 3-4 steps total (fast) |
| Theoretical framing | Spectral orthogonality, mutual information | Banach fixed-point convergence |
| Sub-500M? | Not claimed (uses full VLM backbone) | Explicit 295M target |

**Assessment:** ResVLA validates the residual refinement paradigm for VLA, which is actually good for us — it shows the community considers this direction important. But ResVLA still uses diffusion bridges (slow), not lightweight iterative refinement (fast). Our contribution would be: "ResVLA showed residual refinement works. We show you can do it in 3 steps instead of 50, with convergence guarantees, at sub-500M scale."

### Novelty impact: CONSTRAINED but NOT killed
- We cannot claim "first residual refinement for VLA" — ResVLA owns that
- We CAN claim "first fast iterative residual refinement for sub-500M VLA with convergence guarantees"
- The positioning shifts from "introduce residual refinement" to "make residual refinement practical for edge deployment"

---

## Near-Hit #2: GeCO — "Generative Control as Optimization"

**Paper:** arxiv 2603.17834 (ICML 2026)  
**Authors:** HRH et al.

### What it does
- Time-unconditional flow matching — learns a stationary velocity field
- Adaptive convergence-based inference: exits when the field norm indicates convergence
- Built-in OOD detection via field norm
- Plug-and-play replacement for flow-matching heads

### Does it kill PFM-EE?

**Significantly constrains it.**

GeCO achieves essentially what PFM-EE proposed:
- Adaptive step count based on convergence (not learned gate, but field norm)
- OOD detection built in
- No extra exit gate network needed

Key differences:
| Aspect | GeCO | PFM-EE |
|--------|------|--------|
| Adaptive mechanism | Stationary field norm (intrinsic) | Learned exit gate (extrinsic) |
| OOD detection | Built-in (field norm) | Would need separate module |
| Time conditioning | None (time-unconditional) | Standard flow matching + exit heads |
| Plug-and-play? | Yes | No (requires training exit heads) |

**Assessment:** GeCO is a stronger version of PFM-EE. It achieves adaptive computation without the complexity of learned exit gates, and adds ODD detection for free. PFM-EE would need to demonstrate clear advantages over GeCO, which is hard to justify.

### Novelty impact: KILLED for continuous flow matching
- PFM-EE's core idea (adaptive exit from flow matching) is now claimed by GeCO
- We could pivot to adaptive exit for DISCRETE diffusion (different from GeCO), but DiscreteRTC (see below) covers that too

---

## Near-Hit #3: DiscreteRTC — "Discrete Diffusion Policies are Natural Asynchronous Executors"

**Paper:** arxiv 2604.25050 (published 2 days ago, April 27, 2026)  
**Authors:** Pengcheng Wang, Kaiwen Hong, Chensheng Peng, et al. (UC Berkeley, UIUC, UT Austin, UCLA)

### What it does
- Discrete diffusion policies naturally support early stopping (unmasking)
- Async execution via real-time chunking with discrete diffusion
- 0.7× computation vs generating from scratch
- 50% higher success rate on dynamic pick vs flow-matching RTC

### Relevance
- Shows early exit is natural for discrete diffusion — validates the concept
- But targets async execution (inter-chunk), not single-action speed
- Different from PFM-EE which targets the denoising loop itself

### Novelty impact: Does NOT kill any candidate directly
- Validates early-exit concepts but in a different context (async chunk transitions, not per-action inference speed)

---

## Near-Hit #4: Ada3Drift — "Adaptive Training-Time Drifting for One-Step 3D Visuomotor Policy"

**Paper:** arxiv 2603.11984  
**Authors:** Yixian Zou et al. (Sichuan University, UESTC)

### What it does
- Single-step (1 NFE) generation from 3D point cloud observations
- Training-time drifting field that steers predictions toward expert modes
- Addresses mode-averaging problem of consistency/flow distillation
- 10× fewer function evaluations than diffusion alternatives

### Relevance to DCFH
- Directly addresses the same problem as DCFH (single-step generation)
- But for 3D point-cloud policies, not VLA models
- Uses training-time drifting, not consistency distillation

### Novelty impact: Constrains DCFH slightly
- Ada3Drift shows single-step generation is achievable and SOTA-competitive
- But it's not a VLA, doesn't use a VLM backbone, doesn't use consistency distillation
- DCFH can still claim: "first consistency-distilled single-step VLA action decoder"

---

## Summary: Scoop Verdict

| Candidate | Scoop Status | Verdict |
|-----------|-------------|---------|
| **IRAN** | ResVLA constrains novelty | **ALIVE — needs repositioning** |
| **PFM-EE** | GeCO essentially kills it | **KILLED** |
| **DCFH** | Ada3Drift constrains slightly | **ALIVE — strongest novelty window** |

---

## Revised Recommendation

**DCFH is now the primary candidate.** It has the cleanest novelty window:
- No one has done consistency distillation for VLA action decoders
- Ada3Drift is the closest work but uses a different technique (training-time drifting) and a different domain (3D point-cloud, not VLA)
- The 10× speedup claim is the strongest and most publishable

**IRAN is secondary** — it needs to be repositioned as "making ResVLA's residual refinement practical for edge deployment" rather than introducing residual refinement itself. The convergence theorem is still a unique theoretical contribution.

**PFM-EE is killed** — GeCO does it better with less complexity.
