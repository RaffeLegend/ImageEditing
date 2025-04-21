import torch
from diffusers import FluxControlInpaintPipeline
from diffusers.models.transformers import FluxTransformer2DModel
from transformers import T5EncoderModel
from diffusers.utils import load_image, make_image_grid
from image_gen_aux import DepthPreprocessor # https://github.com/huggingface/image_gen_aux
from PIL import Image
import numpy as np

pipe = FluxControlInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Depth-dev",
    torch_dtype=torch.bfloat16,
)
# use following lines if you have GPU constraints
# ---------------------------------------------------------------
transformer = FluxTransformer2DModel.from_pretrained(
    "sayakpaul/FLUX.1-Depth-dev-nf4", subfolder="transformer", torch_dtype=torch.bfloat16
)
text_encoder_2 = T5EncoderModel.from_pretrained(
    "sayakpaul/FLUX.1-Depth-dev-nf4", subfolder="text_encoder_2", torch_dtype=torch.bfloat16
)
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2
pipe.enable_model_cpu_offload()
# ---------------------------------------------------------------
pipe.to("cuda")

prompt = "a blue robot singing opera with human-like expressions"
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

head_mask = np.zeros_like(image)
head_mask[65:580,300:642] = 255
mask_image = Image.fromarray(head_mask)

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(image)[0].convert("RGB")

output = pipe(
    prompt=prompt,
    image=image,
    control_image=control_image,
    mask_image=mask_image,
    num_inference_steps=30,
    strength=0.9,
    guidance_scale=10.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
make_image_grid([image, control_image, mask_image, output.resize(image.size)], rows=1, cols=4).save("output.png")