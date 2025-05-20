# Generative Video Creation with ConstantParamTanh

## Architecture Overview

(Placeholder Image: A diagram showing the VAE architecture, with a ConstantParamTanh layer inserted after the decoder. Include the Discriminator network.)

The architecture consists of a Variational Autoencoder (VAE) acting as the generator and a Discriminator network. The VAE is trained to generate realistic images, while the Discriminator tries to distinguish between real and generated images. A `ConstantParamTanh` layer is inserted after the decoder to introduce a dynamic normalization effect.

## Network Details

### VAE (Generator)

*   **Architecture:** Encoder (Conv2D, ReLu, MaxPool2D) -> Latent Space (Linear layers for mu/logvar) -> Decoder (ConvTranspose2D, ReLu, Sigmoid).  See `model_definitions.py` for specific layer details.
*   **Input:** Noise vector.
*   **Output:** Generated image (B, C, H, W).

### Discriminator

*   **Architecture:** (The layers are symmetric with the encoder.) See `model_definitions.py` for layer details.
*   **Input:** Real or generated image (B, C, H, W).
*   **Output:** Probability of the input being a real image.

### ConstantParamTanh Layer

*   **Type:** `ConstantParamTanh`
*   **Alpha Value:** There are three modes:
    *   **Single:** A single, learnable scalar value applied to all cells in the layer.
    *   **Matrix**: a matrix of alpha value learnable
    *   **Time Function**: Each cell is the result of a time function.

## Training Data

*   Images of shape (B, C, H, W) = (16, 3, 128, 128).
*   The `ImageFolder` dataset from `torchvision` is used for loading images.

## Training Procedure

1.  The Discriminator is trained to distinguish between real and generated images using the combined Wasserstein loss with gradient penalty.
2.  The VAE is trained to generate realistic images that can fool the Discriminator, also using the combined Wasserstein loss.
3.  The `ConstantParamTanh` layer's `alpha` parameter is learned during the VAE training process.
4.  This approach is done for the 3 modes: Single Value, Matrix and Time Function.

## Usage

1.  Ensure you have PyTorch, torchvision, and tqdm installed.
2.  Organize your image data into a directory structure compatible with `ImageFolder`.
3.  Run the `training.py` script to train the model.
