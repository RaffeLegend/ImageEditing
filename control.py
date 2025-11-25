import torch
import os
import json
from diffusers import FluxControlInpaintPipeline
# from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd3
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1
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
    "sayakpaul/FLUX.1-Depth-dev-nf4", subfolder="transformer", torch_dtype=torch.bfloat16, use_safetensors=True
)
text_encoder_2 = T5EncoderModel.from_pretrained(
    "sayakpaul/FLUX.1-Depth-dev-nf4", subfolder="text_encoder_2", torch_dtype=torch.bfloat16, use_safetensors=True
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

prompt = "You are given an original image and the mask and two captions:   \
- Original Caption: [original caption]                                     \
- Modified Caption: [modified caption]                                     \
Your task is to edit the image on the area covered by the mask based on the semantic difference between the two captions. Only change the visual elements necessary to match the modified caption, while keeping all other elements consistent with the original image."

output_json_path = "new_data.json"  # Path to save the output JSON file
output_json = []

root_dir = "/root/autodl-tmp/"

for idx, item in enumerate(data):
    image_path = item['image_path']
    mask_path = item['mask']
    caption = item['text']
    response = item['headline']

    image = load_image(root_dir + image_path)
    mask_image = load_image(root_dir + "vis_output/" + mask_path)

    mask_np = np.array(mask_image)
    mask_np[mask_np > 10] = 255
    mask_image = Image.fromarray(mask_np)

    control_image = processor(image)[0].convert("RGB")

    filled_prompt = prompt.replace("[original caption]", caption).replace("[modified caption]", response)
    prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(pipe=pipe, prompt=prompt)
    # filled_prompt = get_weighted_text_embeddings_sd3(pipe, prompt=filled_prompt)
    output = pipe(
        # prompt=filled_prompt,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        image=image,
        control_image=control_image,
        mask_image=mask_image,
        num_inference_steps=30,
        strength=0.9,
        guidance_scale=10.0,
        generator=torch.Generator().manual_seed(42),
    ).images[0]
   
    # 生成保存路径（保持原结构）
    base_dir, filename = os.path.split(image_path)
    filename_wo_ext = os.path.splitext(filename)[0]
    new_image_dir = os.path.join("outputs", base_dir)
    os.makedirs(new_image_dir, exist_ok=True)
    output_path = os.path.join(new_image_dir, f"{filename_wo_ext}_output.png")

    # 保存图像网格
    # grid = make_image_grid([image, control_image, mask_image, output.resize(image.size)], rows=1, cols=4)
    output = output.resize(image.size)
    output.save(output_path)

    item['edited'] = output_path
    output_json.append(item)

    # 每处理一条，保存一次json
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print(f"[{idx+1}/{len(data)}] Processed and saved: {output_path}")

print("All samples processed!")
