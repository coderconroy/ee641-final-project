# TL-Diffusion: Conditional Traffic Light Generation using Diffusion Models

This repository contains the source code used to carry of the final project for EE 641 Deep Learning Systems at the University of Southern California.

The repository structure is as follows:

- utils/
    - diffusion.py - Definition of class used to carry out diffusion process.
    - models.py - Definition of UNet models and EMA process used in training of diffusion model.
    - utils.py - Miscellaneuos functions to support sampling and training process.
- diffusion_training.ipynb - Use this file to train the Diffusion model.
- diffusion_sampling.ipynb - Use this file to generate the training samples after a training diffusion model has been saved.
- process_bosch_data.ipynb, process_lisa_data.ipynb, process_s2tld_data.ipynb - Scripts to extract the traffic lights from the various traffic light datasets.
- dcgan.ipynb - Use this file to train the DCGAN model
- fid.ipynb - Use this file to compute the FID score for a set of generated images.

The final model weight  sizes are as follows:

- diffusion_weights.pt - 342 MB
- gan_discriminator_weights.pth - 11 MB
- gan_generator_weights.pth - 14 MB