# Step 1 notes
Overal goal: improve video reconstruction from naural data (calcium imaging, ephys).
Start from transformations from images to firing rates and viceversa.

1. train a neural encoder (image to neurons)
2. use that to train a decoder in 1 of two ways: 
    - **in autoencoder mode**: an image to image autoencoder using the pretrained neural encoder
    - **in neural prediction mode**: neural data goes through decoder to produce an image, the image is then put through the neural encoder and the loss is the orignal neural data and the output activity of the decoder + encoder.

## Steps

### Notation key
Original natural images, as from a dataset like cifar: $I$
It is a matrix of shape (n_images, $w_i$, $h_i$).
The images are grayscale, so only one channel, and their pixel values are normalized between -1 and 1.

Real firing rates: $R$
As calculated from a recording session. It's a matrix of shape (n_neurons, n_timepoints). Every timepoint corresponds to the presentation of an image frame.

Synthetic firing rates: $R_s$

Preferred response filter of a neuron: $G$
It can be a Gabor for instance, 2d squared matrix of shape ($g$ x $g$).
They can have values between -1 and 1.


### Synthetic dataset generation ✅
From images to firing rates.

First, we select a patch of size ($g$ x $g$) from the image $I$. The patch is selected randomly for each neuron, and it is defined by its top-left corner coordinates $(x_j, y_j)$ for neuron $j$.

Patches and filters are then flattened to vectors of size $g * g$.

The dot product between the patch and the filter is computed, resulting in a scalar value $d_j$ for neuron $j$. This value is bound between $-g^2$ and $g^2$. E.g. for a 11x11 filter, the dot product is between -121 and 121.

Then a non-linearity (ELU) is applied to the dot product, and the result is scaled to the maximum firing rate, set to 100Hz by default to simulate realistic firing rates.

Finally, Gaussian noise is added to the firing rate, with a configurable noise level.

Valuels lower than 0 are set to 0.

Here a summary of the equations:
$$
d_j = I_{x_j:x_j+g, y_j:y_j+g} \cdot G_j
$$

$$
R_{s_j} = \frac{\text{ELU}(d_j)}{\max(\text{ELU}(d_j))} \times \text{max\_firing\_rate} + \mathcal{N}(0, \sigma^2)
$$

### Train an encoder ✅
From images to firing rates.

The encoder is a CNN that takes images as input and outputs predicted firing rates for each neuron. We can call it $E(I) = \hat{R_s}$.

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

