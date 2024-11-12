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


variant='diffusion_transformer_image'       # For low resolution
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

#prompt = "shoulder and full head portrait of a beautiful 19 year old girl, brunette, smiling, stunning, highly detailed, glamour lighting, HDR, photorealistic, hyperrealism, octane render, unreal engine"
prompt = "a panda reading a book in a library"

# now support 3 aspect ratios
resolution_dict = {
    #'1:1' : (1024, 1024),
    '1:1' : (2000, 2000),
    '5:3' : (1280, 768),
    '3:5' : (768, 1280),
}

ratio = '1:1'   # 1:1, 5:3, 3:5
#ratio = '3:5'
width, height = resolution_dict[ratio]

with torch.no_grad(), torch.cuda.amp.autocast(enabled=True if model_dtype != 'fp32' else False, dtype=torch_dtype):
    images = model.generate(
        prompt=prompt,
        num_inference_steps=[20, 20, 20],
        height=height,
        width=width,
        temp=1,
        guidance_scale=9.0,        
        output_type="pil",
        save_memory=False, 
    )

images[0].save("./output/output.png")
#import pdb; pdb.set_trace()