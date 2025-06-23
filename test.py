import torch
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from promptdresser.models.unet import UNet2DConditionModel
from promptdresser.models.cloth_encoder import ClothEncoder
from promptdresser.pipelines.sdxl import PromptDresser
from utils.caption import generate_caption
from utils.mask import generate_clothing_mask
from utils.pose import generate_openpose

device = "cuda"
weight_dtype = torch.float16

noise_scheduler = DDPMScheduler.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
    subfolder="scheduler"
)
tokenizer = CLIPTokenizer.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", subfolder="text_encoder")
tokenizer_2 = CLIPTokenizer.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", subfolder="tokenizer_2")
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", subfolder="text_encoder_2")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
unet = UNet2DConditionModel.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", subfolder="unet")
cloth_encoder = ClothEncoder.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")

unet.load_state_dict(torch.load("./checkpoints/VITONHD/model/pytorch_model.bin"))
# cloth_encoder.load_state_dict(torch.load("path_to_your_pretrained_cloth_encoder.pt"))

unet.to(device, dtype=weight_dtype)
vae.to(device, dtype=weight_dtype)
text_encoder.to(device, dtype=weight_dtype)
text_encoder_2.to(device, dtype=weight_dtype)
cloth_encoder.to(device, dtype=weight_dtype)

pipeline = PromptDresser(
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=noise_scheduler,
).to(device, dtype=weight_dtype)

person_image = Image.open("./test/person2.png").convert("RGB")
cloth_image = Image.open("./test/00008_00.jpg").convert("RGB")
mask_image = generate_clothing_mask("./test/person2.png", label=4, output_path="./test/mask2.png", show_result=False)
pose_image = generate_openpose("./test/person2.png", output_image_path="./test/pose2.png", show_result=False)
prompt_person_image = generate_caption("./test/person2.png", device)
prompt_cloth_image = generate_caption("./test/00008_00.jpg", device)


result = pipeline(
    image=person_image,
    mask_image=mask_image,
    pose_image=pose_image,
    cloth_encoder=cloth_encoder,
    cloth_encoder_image=cloth_image,
    prompt=prompt_person_image,
    prompt_clothing=prompt_cloth_image,
    height=1024,
    width=768,
    guidance_scale=2.0,
    guidance_scale_img=4.5,
    guidance_scale_text=7.5,
    num_inference_steps=30,
).images[0]

result.save("./test/output_image.jpg")