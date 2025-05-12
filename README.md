# MoCLIP &nbsp;🚶‍♂️➡️📝  
**Motion-Text CLIP + OTMS / MMMD metrics**  
<sup>*Towards Better Evaluation Metrics for Text-to-Motion Generation*</sup>

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](#license)


MoCLIP is an all-in-one toolkit for **evaluating, analysing and improving text-to-motion models**.  
It ships:

- **MoCLIP encoder** – dual-stream Transformer aligning motions & text (drop-in replacement for CLIP).  
- **OTMS** – *Optimal Transport Matching Score* for global semantic alignment.  
- **MMMD** – *MoCLIP-based Maximum Mean Discrepancy* for distribution fidelity.  
- Plug-and-play scripts, pretrained checkpoints & visualisation utilities.

<details>
<summary><strong>Why MoCLIP?</strong></summary>

| Metric            | Traditional | MoCLIP metrics | Benefit |
|-------------------|-------------|----------------|---------|
| **R-Precision**   | Local only  | Global (OTMS)  | Robust to ambiguous prompts |
| **FID**           | Gaussian-biased | MMMD (non-parametric) | Aligns with human judgement |
| **Speed**         | Slow matrix sqrt | GPU-friendly Sinkhorn | 10× faster on 3 k samples |

<!-- Table values & claims derived from the original paper, Tables 1–2 & Sec. 4.2.1 :contentReference[oaicite:0]{index=0} -->
</details>

---

## ✨ Key Features
* **State-of-the-art retrieval** – 70.5 % Top-1 on HumanML3D (↑ 19 pts vs. baseline). <!-- :contentReference[oaicite:1]{index=1}:contentReference[oaicite:2]{index=2} -->
* **Human-aligned scores** – OTMS / MMMD correlate > 0.80 with human ratings. <!-- :contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4} -->
* **One-line integration** with diffusion, autoregressive or flow models.
* **Lightweight** – single 25 M-param encoder; <150 MB download.

