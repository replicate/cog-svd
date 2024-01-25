# Cog-SDV

This is an implementation of Stability AI's [SDV](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

Run a prediction:

    cog predict -i input_image=@demo.png

## Output

![sample1](output.gif)