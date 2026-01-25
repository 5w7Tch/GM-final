# ML2 Final Project: Generative Models on CIFAR-10

Comparing **NCSN** (Noise Conditional Score Network) and **VAE** (Variational Autoencoder) for unconditional image generation.

## Project Structure

```
ML2_final/
├── src/
│   ├── models/
│   │   ├── ncsn.py      # NCSN & NCSNv2 ✓
│   │   └── vae.py       # VAE (TODO: Partner)
│   ├── losses/
│   │   ├── score_matching.py  # DSM loss ✓
│   │   └── vae_loss.py        # ELBO (TODO: Partner)
│   ├── sampling/
│   │   └── langevin.py  # Annealed Langevin dynamics ✓
│   ├── data/
│   │   └── cifar10.py   # Data loading ✓
│   └── utils/
│       ├── ema.py       # Exponential Moving Average ✓
│       └── visualization.py  # Sample display ✓
├── notebooks/
│   ├── train_ncsn.ipynb # NCSN training (Colab ready) ✓
│   └── train_vae.ipynb  # VAE training (TODO: Partner)
├── requirements.txt
└── README.md
```

## Quick Start (Google Colab)

1. Upload `notebooks/train_ncsn.ipynb` to Colab
2. Update the git clone URL to your repo
3. Run all cells

## Team Division

### Person 1 (NCSN) - DONE ✓
- [x] NCSN model architecture (RefineNet-based)
- [x] Denoising Score Matching loss
- [x] Annealed Langevin Dynamics sampling
- [x] Training notebook

### Person 2 (VAE) - TODO
- [ ] Implement `src/models/vae.py`:
  - Encoder: Conv layers → (μ, log σ²)
  - Decoder: Deconv layers → x_reconstructed
  - Reparameterization trick
- [ ] Implement `src/losses/vae_loss.py`:
  - Reconstruction loss (MSE)
  - KL divergence
- [ ] Complete `notebooks/train_vae.ipynb`

## Key Formulas

### NCSN
- **Score**: s_θ(x, σ) ≈ ∇_x log p_σ(x)
- **DSM Loss**: ||s_θ(x̃) + z/σ||² where x̃ = x + σz
- **Sampling**: x_{t+1} = x_t + ε·s_θ(x_t) + √(2ε)·z

### VAE
- **ELBO**: log p(x) ≥ E[log p(x|z)] - KL(q(z|x) || p(z))
- **Reparameterization**: z = μ + σ ⊙ ε, ε ~ N(0, I)

## References

- NCSN: [Song & Ermon, NeurIPS 2019](https://arxiv.org/abs/1907.05600)
- VAE: [Kingma & Welling, 2014](https://arxiv.org/abs/1312.6114)
