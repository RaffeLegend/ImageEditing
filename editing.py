from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from transformers import AutoTokenizer, AutoModel
import requests
from io import BytesIO

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 FLUX 模型
# 注意：FLUX 官方仓库是 PAIR/flux，如果已支持 image editing 任务，替换这里
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)

# 加载图像和 mask
def load_image(url_or_path):
    if url_or_path.startswith("http"):
        response = requests.get(url_or_path)
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(url_or_path).convert("RGB")

original_image = load_image("original.jpg")
ref_image = load_image("reference.jpg")
mask_image = load_image("mask.png")  # 黑白图，白色部分为编辑区域
instruction = "Make the sky in the original image look like the reference image"

# 尺寸统一
original_image = original_image.resize((512, 512))
ref_image = ref_image.resize((512, 512))
mask_image = mask_image.resize((512, 512))

# 推理
edited_image = pipe(
    image=original_image,
    reference_image=ref_image,
    mask_image=mask_image,
    prompt=instruction
).images[0]

# 保存结果
edited_image.save("edited_output.png")
edited_image.show()