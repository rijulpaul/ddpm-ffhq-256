from dataclasses import dataclass


@dataclass
class TrainingConfig:
    image_size = 128
    batch_size = 32
    eval_batch_size = 8
    num_epochs = 300
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 2000
    save_image_epochs = 10
    save_model_epochs = 2
    mixed_precision = "fp16"
    seed = 0

    data_dir = "dataset"
    output_dir = "output"

    upload_to_hf = False
    repo_id = None

    device = "cuda"
