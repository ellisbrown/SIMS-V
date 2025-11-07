<div align="center">


<!-- TITLE -->
# SIMS-V: Simulated Instruction-Tuning for Spatial Video Understanding

<!-- BADGES -->

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv:2511.04668-b31b1b.svg?style&logo=arXiv)](http://arxiv.org/abs/2511.04668)
[![arXiv](https://img.shields.io/badge/📄_PDF-SIMS--V-FDDEB3.svg)](https://arxiv.org/pdf/2511.04668)
[![Project](https://img.shields.io/badge/🌎_Web-SIMS--V-blue.svg)](https://ellisbrown.github.io/sims-v/)
[![Home](https://img.shields.io/badge/HF-SIMS--VSI-FED123.svg?style&logo=HuggingFace)](https://hf.co/datasets/ellisbrown/SIMS-VSI)

</div>


<!-- DESCRIPTION -->
## Abstract
Despite impressive high-level video comprehension, multimodal language models struggle with spatial reasoning across time and space. While current spatial training approaches rely on real-world video data, obtaining diverse footage with precise spatial annotations remains a bottleneck. To alleviate this bottleneck, we present SIMS-V—a systematic data-generation framework that leverages the privileged information of 3D simulators to create spatially-rich video training data for multimodal language models. Using this framework, we investigate which properties of simulated data drive effective real-world transfer through systematic ablations of question types, mixes, and scales. We identify a minimal set of three question categories (metric measurement, perspective-dependent reasoning, and temporal tracking) that prove most effective for developing transferable spatial intelligence, outperforming comprehensive coverage despite using fewer question types. These insights enable highly efficient training: our 7B-parameter video LLM fine-tuned on just 25K simulated examples outperforms the larger 72B baseline and achieves competitive performance with proprietary models on rigorous real-world spatial reasoning benchmarks. Our approach demonstrates robust generalization, maintaining performance on general video understanding while showing substantial improvements on embodied and real-world spatial tasks.



## Code
Coming soon!


<!-- CITATION -->
## Citation

```bibtex
@article{brown2025simsv,
  title = {{SIMS-V}: Simulated Instruction-Tuning for Spatial Video Understanding},
  author = {Brown, Ellis and Ray, Arijit and Krishna, Ranjay and Girshick, Ross and Fergus, Rob and Xie, Saining},
  journal = {arXiv preprint arXiv:2511.04668},
  year = {2025},
}
```
 