# VLA: Variational Linear Attention

Unofficial PyTorch implementation of **Variational Linear Attention (VLA)**, based on *[Variational Linear Attention: Stable Associative Memory for Long-Context Transformers (Pandey & Singh, 2026)](https://arxiv.org/html/2605.11196v1)*.

### Overview

This repository provides:

* **`vla_sequential_forward`**: Reference sequential implementation (Algorithm 1) featuring adaptive Sherman-Morrison updates for stable memory retention.
* **`vla_semi_parallel_s_update`**: Associative scan-based structure for investigating latent memory dynamics.

### Citation

```bibtex
@article{pandey2026variational,
  title={Variational Linear Attention: Stable Associative Memory for Long-Context Transformers},
  author={Pandey, Vishal and Singh, Gopal},
  journal={arXiv preprint arXiv:2605.11196},
  year={2026}
}

```

*Disclaimer: This is an educational and experimental implementation not affiliated with the original authors.*