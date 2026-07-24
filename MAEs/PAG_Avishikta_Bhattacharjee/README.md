# Physics-Aware Gated LorentzParT for Jet Reconstruction

This repository contains the implementation of a **Physics-Aware Gated Lorentz Particle Transformer (LorentzParT)** designed for self-supervised pre-training and jet reconstruction on high-energy physics datasets like **JetClass**. 

The core contribution of this work is a custom attention block that injects a **normalized global invariant mass bias ($m^2$)** alongside a **gated pairwise interaction matrix ($U$)** directly into the attention mechanism, enforcing fundamental Lorentz invariance and relativistic conservation laws.

---


## Description

Standard Transformers evaluate token-to-token relationships purely through statistical dot-product attention ($QK^T$). In high-energy physics, this often leads to attention maps that violate basic physical constraints like energy-momentum conservation or relativistic invariance.

### 1. Reconstructing Particle Kinematics
Each constituent particle in a jet is defined by its 4-momentum $(p_T, \eta, \phi, E)$. These cylindrical collider coordinates convert to Cartesian 3D momentum $(p_x, p_y, p_z)$ via:

$$p_x = p_T \cos(\phi), \quad p_y = p_T \sin(\phi), \quad p_z = p_T \sinh(\eta), \quad p = p_T \cosh(\eta)$$

### 2. Global Invariant Mass Scalar ($m^2$)
For a jet of $N$ particles, the total invariant mass squared ($m^2$) is computed across aggregate energy and momentum sums:

$$m^2 = \left(\sum_{i=1}^{N} E_i\right)^2 - \left\| \sum_{i=1}^{N} \vec{p}_i \right\|_2^2$$

To ensure numerical stability across varying energy scales, $m^2$ is feature-scaled to form a **normalized invariant mass bias**.



## 🛠️ Architecture Overview

The model uses the encoder of LorentzParT embedded within a Variational Autoencoder (VAE) setup:
1. **Masking:** Random constituent particles are masked during training.
2. **Encoder:** Processes unmasked tokens through LorentzParT blocks augmented with the **Physics-Aware Gating (PAG)** layer.
3. **Decoder:** Reconstructs the 4 constituent properties ($p_T, \eta, \phi, E$) of the masked particles.

---

## 📊 Evaluation & Results

Will add in the final submission (yet to upload)
