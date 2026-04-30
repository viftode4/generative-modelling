# Glossary

## Classifier-free guidance

A diffusion sampling technique that combines conditional and unconditional model predictions to steer generation toward a prompt.

## Denoising

The reverse diffusion process step that moves a noisy latent toward a cleaner sample.

## Diffusion inversion

A process that maps a cleaner sample or latent back toward a noisier state, often used for editing, reconstruction, or analysis of diffusion trajectories.

## Guidance gap

TODO: Define precisely from the Z-Sampling paper equations. Current rough understanding: a difference between guided denoising and guided inversion behavior that can carry prompt-related semantic information.

## Z-Sampling

Zigzag Diffusion Sampling, the method studied in this project. It alternates denoising and inversion to self-improve diffusion generation during sampling.
