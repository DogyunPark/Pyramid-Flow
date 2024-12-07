import math
import sys
from typing import Iterable
import os

import torch
import torch.nn as nn
import accelerate
from einops import rearrange
from .utils import MetricLogger, SmoothedValue
from diffusers.utils import export_to_video
import random
import torchvision.transforms as transforms
from PIL import Image


def save_video_batch_to_png(video_tensor, output_dir, prefix="video"):
    """
    Save each frame of each video in a batch as PNG images.

    Args:
        video_tensor (torch.Tensor): The video tensor of shape (B, C, T, H, W),
                                     where B is the batch size, C is the number of channels,
                                     T is the number of frames, H is the height, and W is the width.
        output_dir (str): The directory where the PNG images will be saved.
        prefix (str): The prefix for the saved image filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()

    for b in range(video_tensor.size(0)):  # Iterate over batch
        batch_dir = os.path.join(output_dir, f"{prefix}_{b}")
        os.makedirs(batch_dir, exist_ok=True)
        for t in range(video_tensor.size(2)):  # Iterate over frames
            frame = video_tensor[b, :, t, :, :]
            frame = (frame + 1.0) / 2.0
            image = to_pil(frame)
            image.save(os.path.join(batch_dir, f"frame_{t:04d}.png"))

def update_ema_for_dit(model, model_ema, accelerator, decay):
    """Apply exponential moving average update.

    The  weights are updated in-place as follow:
    w_ema = w_ema * decay + (1 - decay) * w
    Args:
        model: active model that is being optimized
        model_ema: running average model
        decay: exponential decay parameter
    """
    with torch.no_grad():
        msd = accelerator.get_state_dict(model)
        for k, ema_v in model_ema.state_dict().items():
            if k in msd:
                model_v = msd[k].detach().to(ema_v.device, dtype=ema_v.dtype)
                ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)


def get_decay(optimization_step: int, ema_decay: float) -> float:
    """
    Compute the decay factor for the exponential moving average.
    """
    step = max(0, optimization_step - 1)

    if step <= 0:
        return 0.0

    cur_decay_value = (1 + step) / (10 + step)
    cur_decay_value = min(cur_decay_value, ema_decay)
    cur_decay_value = max(cur_decay_value, 0.0)

    return cur_decay_value


def train_one_epoch_with_fsdp(
    runner,
    model_ema: torch.nn.Module,
    accelerator: accelerate.Accelerator,
    model_dtype: str,
    data_loader: Iterable, 
    optimizer: torch.optim.Optimizer,
    lr_schedule_values,
    device: torch.device, 
    epoch: int, 
    clip_grad: float = 1.0,
    start_steps=None,
    args=None,
    print_freq=20,
    iters_per_epoch=2000,
    ema_decay=0.9999,
    use_temporal_pyramid=True,
    validation_prompt=None,
    validation_image=None,
    save_intermediate_latents=False,
):
    runner.dit.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0.0

    topil = transforms.ToPILImage()
    print("Start training epoch {}, {} iters per inner epoch. Training dtype {}".format(epoch, iters_per_epoch, model_dtype))

    for step in metric_logger.log_every(range(iters_per_epoch), print_freq, header):
        if step >= iters_per_epoch:
            break

        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule_values[start_steps] * param_group.get("lr_scale", 1.0)

        for _ in range(args.gradient_accumulation_steps):

            with accelerator.accumulate(runner.dit):
                # To fetch the data sample and Move the input to device
                samples = next(data_loader)
                video =  samples['video'].to(accelerator.device)
                text = samples['text']

                import pdb; pdb.set_trace()
                save_video_batch_to_png(video, "./output/video_sample", prefix="train")

                loss, log_loss = runner(video, text, identifier=None, accelerator=accelerator)

                # Check if the loss is nan
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value), force=True)
                    sys.exit(1)

                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()

                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)

                # clip the gradient
                if accelerator.sync_gradients:
                    params_to_clip = runner.dit.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, clip_grad)
                
                # To deal with the abnormal data point
                if train_loss >= 2.0:
                    print(f"The ERROR data sample, finding extreme high loss {train_loss}, skip updating the parameters", force=True)
                    # zero out the gradient, do not update
                    optimizer.zero_grad()
                    train_loss = 0.001    # fix the loss for logging
                else:
                    optimizer.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                # Update every 100 steps
                if model_ema is not None and start_steps % 100 == 0:
                    # cur_ema_decay = get_decay(start_steps, ema_decay)
                    cur_ema_decay = ema_decay
                    update_ema_for_dit(runner.dit, model_ema, accelerator, decay=cur_ema_decay)

                start_steps += 1

                # Report to tensorboard
                accelerator.log({"train_loss": train_loss}, step=start_steps)
                metric_logger.update(loss=train_loss)

                train_loss = 0.0

                min_lr = 10.
                max_lr = 0.
                for group in optimizer.param_groups:
                    min_lr = min(min_lr, group["lr"])
                    max_lr = max(max_lr, group["lr"])

                metric_logger.update(lr=max_lr)
                metric_logger.update(min_lr=min_lr)
                weight_decay_value = None
                for group in optimizer.param_groups:
                    if group["weight_decay"] > 0:
                        weight_decay_value = group["weight_decay"]
                metric_logger.update(weight_decay=weight_decay_value)
                metric_logger.update(grad_norm=grad_norm)


        if (step+1) % 1000 == 0:
            accelerator.wait_for_everyone()
            if args.output_dir:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}-{epoch}")
                accelerator.save_state(save_path, safe_serialization=False)

        accelerator.wait_for_everyone()

        validation_prompt = ["a Emu, focused yet playful, ready for a competitive matchup, photorealistic quality with cartoon vibes","real beautiful woman, Chinese", "marvel movie character, iron man, dress up to match movie character, full body photo, American apartment, lying down, life in distress, messy, lost hope, food, wine, hd, 8k, real, reality, super detail, 8k post photo manipulation, real photo"]
        # Generate the video for validation
        if step % 20 == 0:
            runner.dit.eval()
            # print("Generating the video for text: {}".format(text[0]))
            # image = runner.generate_laplacian_video(
            #     prompt=text[0],
            #     input_image=video[:1, :, :1],
            #     num_inference_steps=[10, 10, 10],
            #     output_type="pil",
            #     save_memory=True,
            #     guidance_scale=9.0,
            #     generation_height=512,
            #     generation_width=512
            # )
            # if save_intermediate_latents:
            #     for i_img, img in enumerate(image):
            #         export_to_video(img, "./output/text_to_video_sample-{}epoch-train-{}.mp4".format(epoch, i_img), fps=24)
            # else:
            #     export_to_video(image, "./output/text_to_video_sample-{}epoch-train.mp4".format(epoch), fps=24)
            assert validation_prompt is not None and validation_image is not None
            for num_image in range(3):
                prompt = validation_prompt[num_image]
                img = validation_image[num_image][:,0].to(accelerator.device)
                img = img.unsqueeze(0).unsqueeze(2)

                image = runner.generate_laplacian_video(
                    prompt=prompt,
                    input_image=img,
                    num_inference_steps=[10, 10, 10],
                    output_type="pil",
                    save_memory=True,
                    guidance_scale=9.0,
                    generation_height=512,
                    generation_width=512
                )
                if save_intermediate_latents:
                    for i_img, img in enumerate(image):
                        export_to_video(img, "./output/text_to_video_sample-{}epoch-{}-{}.mp4".format(epoch, num_image, i_img), fps=24)
                else:
                    export_to_video(image, "./output/text_to_video_sample-{}epoch-{}.mp4".format(epoch, num_image), fps=24)
            print("Generated video for {} step/{} epoch".format(step, epoch))
            accelerator.wait_for_everyone()
            runner.dit.train()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}