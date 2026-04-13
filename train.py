import argparse
from src.config import TrainingConfig
from src.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM model")

    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset")
    parser.add_argument("--upload-to-hf",action="store_true", help="Upload checkpoint to Hugging Face")
    parser.add_argument("--repo-id", type=str, default=None, help="HF Repo id to upload checkpoint to")
    parser.add_argument("--output", type=str, default=None, help="Checkpoint/output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = TrainingConfig()

    # Override only if provided
    if args.dataset:
        config.data_dir = args.dataset

    if bool(args.upload_to_hf) != bool(args.repo_id):
        raise ValueError("Uploading to HuggingFace requires both --upload-to-hf and --repo-id")
    else:
        config.upload_to_hf = True
        config.repo_id = args.repo_id

    if args.output:
        config.output_dir = args.output

    train(config, checkpoint_path = args.checkpoint)
