# Cog-SDV

This is an implementation of Stability AI's [SDV](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

First download the checkpoint:

    cog run script/download-weights

Then run aprediction:

    cog predict -i prompt="a photo of TOK"

## Output

![sample1](output.gif)