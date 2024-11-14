import os
import json
import torch
import numpy as np
import PIL
from PIL import Image
#from IPython.display import HTML
from pyramid_dit import PyramidDiTForVideoGeneration
#from IPython.display import Image as ipython_image
from diffusers.utils import load_image, export_to_video, export_to_gif


variant='diffusion_transformer_384p'       # For low resolution
model_name = "pyramid_flux"

model_path = "./output/pyramid-flow-miniflux"   # The downloaded checkpoint dir
model_dtype = 'bf16'

device_id = 0
torch.cuda.set_device(device_id)


model = PyramidDiTForVideoGeneration(
    model_path,
    model_dtype,
    model_name=model_name,
    model_variant=variant,
)

model.vae.to("cuda")
model.dit.to("cuda")
model.text_encoder.to("cuda")

model.vae.enable_tiling()

if model_dtype == "bf16":
    torch_dtype = torch.bfloat16 
elif model_dtype == "fp16":
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

prompt = 'a woman is walking'
#prompt = "A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors"

# used for 384p model variant
width = 640
height = 640

# used for 768p model variant
# width = 1280
# height = 768

temp = 8   # temp in [1, 31] <=> frame in [1, 241] <=> duration in [0, 10s]
# For the 384p version, only supports maximum 5s generation (temp = 16)

with torch.no_grad(), torch.cuda.amp.autocast(enabled=True if model_dtype != 'fp32' else False, dtype=torch_dtype):
    frames = model.generate(
        prompt=prompt,
        num_inference_steps=[20, 20, 20],
        video_num_inference_steps=[10, 10, 10],
        height=height,
        width=width,
        temp=temp,
        guidance_scale=7.0,         # The guidance for the first frame, set it to 7 for 384p variant
        video_guidance_scale=5.0,   # The guidance for the other video latent
        output_type="pil",
        save_memory=False,           # If you have enough GPU memory, set it to `False` to improve vae decoding speed
    )

#frames[0].save("./output/output.png")
export_to_video(frames, "./output/text_to_video_sample.mp4", fps=24)
#import pdb; pdb.set_trace()