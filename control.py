import torch
import json
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
processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")

input_path = "data.json"  # Path to your JSON file
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

prompt = "You are given an original image and the mask and two captions:
- Original Caption: [original caption]
- Modified Caption: [modified caption]
Your task is to edit the image on the area covered by the mask based on the semantic difference between the two captions. Only change the visual elements necessary to match the modified caption, while keeping all other elements consistent with the original image."

for item in data:
    image_path = item['image_path']
    mask_path = item['mask']
    caption = item['text']
    response = item['response']

    image = load_image(image_path)
    mask_image = load_image(mask_path)

    head_mask = np.zeros_like(image)
    head_mask[65:580,300:642] = 255
    mask_image = Image.fromarray(head_mask)

    control_image = processor(image)[0].convert("RGB")

    filled_prompt = prompt.replace("[original caption]", text).replace("[modified caption]", response)

    output = pipe(
        prompt=filled_prompt,
        image=image,
        control_image=control_image,
        mask_image=mask_image,
        num_inference_steps=30,
        strength=0.9,
        guidance_scale=10.0,
        generator=torch.Generator().manual_seed(42),
    ).images[0]
   
    item_name = os.path.basename(image_path).split(".")[0]
    output_path = f"output_{item_name}.png"
    make_image_grid([image, control_image, mask_image, output.resize(image.size)], rows=1, cols=4).save(output_path)
