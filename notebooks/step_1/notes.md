# Step 1 notes
Overal goal: improve video reconstruction from naural data (calcium imaging, ephys).

# Synthetic dataset playground
Start from transformations from images to firing rates and viceversa.

### Notation key
Original natural images, as from a dataset like cifar: $I$
It is a matrix of shape (width, height, channels).

Real firing rates: $r$
As calculated from a recording session.

Synthetic firing rates: $r_s$

Preferred response filter of a neuron: $g$
It can be a gabor for instance, 2d matrix (width, height).

## Steps

### Synthetic dataset generation ✅
From images to firing rates.


### Train an encoder ✅
From images to firing rates.

### Train a decoder
1. From synthetic firing rates to images.
2. From predicted firing rates to images.

### Train an autoencoder
1. from firing rates to firing rates using trained encoder.
Rationale: data augmentation

2. from images to images, with latent space of a similar dimensionality to the neruons.

### Generate MEI
From firing rates to images, using trained encoder.
Rationale: proof that the encoder is learning a meaningful representation.

