# ML2 Final Project: Generative Models on CIFAR-10

Comparing **NCSN** (Noise Conditional Score Network) and **VAE** (Variational Autoencoder) for unconditional image generation.

## Project Structure

```
GM-final/
├── src/
│   ├── models/
│   │   ├── ncsn.py      # NCSN & NCSNv2 ✓
│   │   └── vae.py       # VAE (TODO)
│   ├── losses/
│   │   ├── score_matching.py  # DSM loss ✓
│   │   └── vae_loss.py        # ELBO (TODO)
│   ├── sampling/
│   │   └── langevin.py  # Annealed Langevin dynamics ✓
│   ├── data/
│   │   └── cifar10.py   # Data loading ✓
│   └── utils/
│       ├── ema.py       # Exponential Moving Average ✓
│       └── visualization.py  # Sample display ✓
├── notebooks/
│   ├── train_ncsn.ipynb # NCSN training (Colab ready) ✓
│   └── train_vae.ipynb  # VAE training (TODO)
├── requirements.txt
└── README.md
```

## Quick Start (Google Colab)

1. Upload `notebooks/train_ncsn.ipynb` to Colab
2. Update the git clone URL to your repo
3. Run all cells

## Team Division

### (NCSN) - DONE ✓
- [x] NCSN model architecture (RefineNet-based)
- [x] Denoising Score Matching loss
- [x] Annealed Langevin Dynamics sampling
- [x] Training notebook

### (VAE) - TODO
- [ ] Implement `src/models/vae.py`:
  - Encoder: Conv layers → (μ, log σ²)
  - Decoder: Deconv layers → x_reconstructed
  - Reparameterization trick
- [ ] Implement `src/losses/vae_loss.py`:
  - Reconstruction loss (MSE)
  - KL divergence
- [ ] Complete `notebooks/train_vae.ipynb`


## References

- NCSN: [Song & Ermon, NeurIPS 2019](https://arxiv.org/abs/1907.05600)
- VAE: [Kingma & Welling, 2014](https://arxiv.org/abs/1312.6114)
