import os
import shutil
import kagglehub
import torch
from torchvision import transforms
from datasets import load_dataset


def ensure_dataset(config):

    if os.path.exists(config.data_dir) and len(os.listdir(config.data_dir)) > 0:
        print(f"Using existing dataset at: {config.data_dir}")
        return config.data_dir

    print("Dataset not found. Downloading from Kaggle...")

    download_path = kagglehub.dataset_download("rijulpaul/ffhq-dataset-128x128")

    print(f"Downloaded to: {download_path}")

    os.makedirs(config.data_dir, exist_ok=True)

    for item in os.listdir(download_path):
        src = os.path.join(download_path, item)
        dst = os.path.join(config.data_dir, item)

        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print(f"Dataset prepared at: {config.data_dir}")

    return config.data_dir


def get_dataloader(config):
    data_path = ensure_dataset(config)

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(img.convert("RGB")) for img in examples["image"]]
        return {"images": images}

    dataset = load_dataset("imagefolder", data_dir=data_path, drop_labels=True)["train"]
    dataset.set_transform(transform)

    print(f"Loaded {len(dataset)} images")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataloader
