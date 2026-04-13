from diffusers import DDPMScheduler

def get_noise_scheduler():
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2"
    )
