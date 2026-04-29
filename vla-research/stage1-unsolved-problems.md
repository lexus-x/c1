# Stage 1 — Unsolved Problems in Sub-1B VLA

**Date:** 2026-04-29  
**Goal:** Find a novel, publishable direction for a sub-500M VLA model

## Current Sub-1B Landscape

| Model | Params | Backbone | Action Head | Key Limitation |
|-------|--------|----------|-------------|----------------|
| SmolVLA | 450M | SmolVLM-256M | Flow matching | Still requires large-scale pre-training data |
| TinyVLA | ~1B | Trained from scratch | Diffusion + LoRA | Not truly sub-500M; no cross-embodiment |
| MiniVLA | ~1B | Qwen2-0.5B | VQ-VAE tokenizer | Fixed action vocabulary, poor multi-modal |
| LLaVA-VLA | lightweight | LLaVA compact | Action chunking | No mobile manipulation generalization |
| Edge-VLA | ~1B | Qwen2-0.5B | — | Not published results on standard benchmarks |

## Top 5 Unsolved Problems

Ranked by: citation pressure × absence of solution × sub-500M tractability

### #1 — Action Representation Bottleneck

**The gap:** Every sub-1B VLA uses a suboptimal action head. Autoregressive tokenizers (OpenVLA-style) are slow and lose continuous precision. Diffusion heads (SmolVLA, TinyVLA) add 10-100 denoising steps per action. Flow matching is better but still 10-50 steps. No sub-500M model has demonstrated a single-pass continuous action decoder that matches diffusion quality.

**Evidence:**
- ICLR 2026 had 4 concurrent discrete diffusion VLA papers — all targeting action decoding speed
- DFM-VLA (arxiv 2603.26320) identifies the "irreversible commitment problem" in AR and discrete diffusion decoding
- The Efficient VLA survey (arxiv 2510.24795) lists "real-time incompatibility" as bottleneck #1

**Why sub-500M tractable:** The action decoder is orthogonal to the backbone size. A 250M backbone + a novel 50M action head = 300M total. The innovation is in the decoder, not scaling.

### #2 — No Sub-500M VLA Does Embodied Chain-of-Thought Reasoning

**The gap:** ECoT requires generating intermediate reasoning tokens before actions. Current ECoT implementations use autoregressive VLMs → extremely slow. Discrete diffusion can do parallel generation but nobody has combined discrete diffusion reasoning with continuous action generation in a sub-500M model.

**Evidence:**
- ICLR 2026 trend: "Reasoning VLAs and ECoT" — all implementations are 3B+ or extremely slow
- dVLA shows ECoT + discrete diffusion works but is still large-scale

### #3 — Cross-Embodiment Without Pre-Training

**The gap:** No sub-500M VLA has demonstrated zero-shot transfer across embodiments without massive pre-training. OpenVLA requires 21,500 A100-GPU hours. π0 requires 10,000+ hours of robotic trajectories.

### #4 — Visual Token Explosion

**The gap:** Multi-view + proprioception + language = 500+ tokens for a 250M backbone. Nobody has principled token compression that preserves task-critical spatial information.

### #5 — No Sub-500M VLA Has Real Safety Guarantees

**The gap:** No sub-500M model has built-in confidence estimation, OOD detection, or safe fallback mechanisms.

## Decision

**Focus: Problem #1 — Action Representation Bottleneck**

Reasons:
- Clear evaluation metrics: latency, success rate, trajectory smoothness
- Independent of backbone scale
- Active research pressure (multiple recent papers targeting decoding)
- Feasible within sub-500M constraint

## References

- Survey: "An Anatomy of VLA Models" — arxiv 2512.11362
- Survey: "A Survey on Efficient VLAs" — arxiv 2510.24795
- Blog: "State of VLA Research at ICLR 2026" — mbreuss.github.io
- Paper: "Rethinking Practicality of VLA" (LLaVA-VLA) — arxiv 2602.22663
- Paper: "DySL-VLA" — arxiv 2602.22896
- Paper: "DFM-VLA" — arxiv 2603.26320
- Paper: "Scaling Verification" (CoVer-VLA) — arxiv 2602.12281
