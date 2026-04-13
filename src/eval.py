import os
import torch
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid

def evaluate(config, epoch, pipeline, model, ema):
    ema.apply_shadow(model)

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    images = pipeline(
        batch_size=config.eval_batch_size,
        num_inference_steps=50,
        generator=torch.Generator(device=config.device).manual_seed(42),
    ).images

    ema.restore(model)

    image_grid = make_image_grid(images, rows=2, cols=4)

    os.makedirs(f"{config.output_dir}/samples", exist_ok=True)
    image_grid.save(f"{config.output_dir}/samples/{epoch:04d}.png")
