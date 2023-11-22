# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import cv2
import math
import torch
import numpy as np
from PIL import Image
from glob import glob
from typing import Optional
from omegaconf import OmegaConf
from einops import rearrange, repeat
from torchvision.transforms import ToTensor
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from sizing_strategy import SizingStrategy
from weights_downloader import WeightsDownloader

"""Exported from stability/ai generative-models """


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device)
    else:
        model = instantiate_from_config(config.model).to(device)

    # FP16
    model.model.half()
    model.eval()
    return model


SVD_MODEL_CACHE = "./checkpoints"
SVD_URL = "https://weights.replicate.delivery/default/svd/svd_and_svd_xt.tar"

SVD_DEFAULT_FRAMES = 14
SVD_DEFAULT_STEPS = 25

SVD_XT_DEFAULT_FRAMES = 25
SVD_XT_DEFAULT_STEPS = 30

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.sizing_strategy = SizingStrategy()
        WeightsDownloader.download_if_not_exists(SVD_URL, SVD_MODEL_CACHE)

        self.svd_model = load_model(
            "svd.yaml",
            "cuda",
            SVD_DEFAULT_FRAMES,
            SVD_DEFAULT_STEPS,
        )

        self.svd_xt_model = load_model(
            "svd_xt.yaml",
            "cuda",
            SVD_XT_DEFAULT_FRAMES,
            SVD_XT_DEFAULT_STEPS,
        )

        # self.model = torch.load("./weights.pth")
        # TODO: cache & download open_clip_pytorch_model.bin here

    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        video_length: str = Input(
            description="Use svd or svd_xt",
            choices=[
                "14_frames_with_svd",
                "25_frames_with_svd_xt",
            ],
            default="14_frames_with_svd",
        ),
        sizing_strategy: str = Input(
            description="Decide how to resize the input image",
            choices=[
                "maintain_aspect_ratio",
                "crop_to_16_9",
                "use_image_dimensions",
            ],
            default="maintain_aspect_ratio",
        ),
        frames_per_second: int = Input(description="Frames per second", default=6, ge=5, le=30),
        motion_bucket_id: int = Input(
            description="Increase overall motion in the generated video", default=127, ge=1, le=255
        ),
        cond_aug: float = Input(description="conditional aug", default=0.02),
        decoding_t: int = Input(description="Number of frames to decode at a time", default=7),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        # Remove individual frame images
        output_folder: Optional[str] = "output/"
        for file_name in glob(os.path.join(output_folder, "*.png")):
            os.remove(file_name)

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = self.sizing_strategy.apply(sizing_strategy, input_image)

        device = "cuda"
        print("Set consts")

        if video_length == "14_frames_with_svd":
            model = self.svd_model
            num_frames = SVD_DEFAULT_FRAMES
        else:
            model = self.svd_xt_model
            num_frames = SVD_XT_DEFAULT_FRAMES

        print("Loaded model")
        torch.manual_seed(seed)

        output_path = None

        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = ToTensor()(image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024):
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if frames_per_second < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if frames_per_second > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = frames_per_second
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                os.makedirs(output_folder, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder, "*.mp4")))
                video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
                output_path = video_path

                samples = embed_watermark(samples)
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                # Save frames as individual images
                for i, frame in enumerate(vid):
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        os.path.join(output_folder, f"frame_{i:06d}.png"), frame
                    )

                # Use ffmpeg to create video from images
                os.system(
                    f"ffmpeg -r {frames_per_second + 1} -i {output_folder}/frame_%06d.png -c:v libx264 -vf 'fps={frames_per_second + 1},format=yuv420p' {video_path}"
                )

                # Remove individual frame images
                for file_name in glob(os.path.join(output_folder, "*.png")):
                    os.remove(file_name)

        return Path(output_path)
