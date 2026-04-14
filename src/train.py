import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
import bitsandbytes as bnb

from .dataset import get_dataloader
from .model import get_model
from .ema import EMA
from .eval import evaluate
from .noise_scheduler import get_noise_scheduler
from .utils import upload_to_hf


def train(config, checkpoint_path):

    start_epoch = 0
    dataloader = get_dataloader(config)
    model = get_model(config)
    ema = EMA(model)

    noise_scheduler = get_noise_scheduler()

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(dataloader) * config.num_epochs,
    )

    scaler = torch.amp.GradScaler("cuda")

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=config.device)

        model = get_model(config,checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        ema.shadow = checkpoint["ema"]

        scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = checkpoint["epoch"] + 1

        print(f"Continuing training from checkpoint: Epoch {start_epoch}")

    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            unit="it",
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for step, batch in enumerate(dataloader):
            clean_images = batch["images"].to(config.device)

            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=config.device,
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with torch.amp.autocast("cuda"):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()
            ema.update(model)

            progress_bar.update(1)

        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

        if epoch % config.save_image_epochs == 0:
            evaluate(config, epoch, pipeline, model, ema)

            if (config.upload_to_hf):
                upload_to_hf(config.repo_id,config.output_dir, commit_msg=f"Evaluation Images: Epoch :{epoch}")

        if epoch % config.save_model_epochs == 0:
            os.makedirs(config.output_dir, exist_ok=True)

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "ema": ema.shadow,
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                },
                f"{config.output_dir}/checkpoint.pt",
            )

            pipeline.save_pretrained(f"{config.output_dir}/pipeline")

            if (config.upload_to_hf):
                upload_to_hf(config.repo_id,config.output_dir,commit_msg=f"Training Checkpoint: Epoch {epoch}")
