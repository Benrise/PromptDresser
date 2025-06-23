import os
import torch
import gradio as gr
import tempfile
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

from promptdresser.models.unet import UNet2DConditionModel
from promptdresser.models.cloth_encoder import ClothEncoder
from promptdresser.pipelines.sdxl import PromptDresser
from lib.caption import generate_caption
from lib.mask import generate_clothing_mask
from lib.pose import generate_openpose


device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.float16 if device == "cuda" else torch.float32

def load_models():
    print("⚙️ Загрузка моделей...")
    
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

    unet_checkpoint_path = hf_hub_download(
        repo_id="Benrise/VITON-HD",
        filename="VITONHD/model/pytorch_model.bin",
        cache_dir="checkpoints"
    )
    unet.load_state_dict(torch.load(unet_checkpoint_path))
    
    models = {
        "unet": unet.to(device, dtype=weight_dtype),
        "vae": vae.to(device, dtype=weight_dtype),
        "text_encoder": text_encoder.to(device, dtype=weight_dtype),
        "text_encoder_2": text_encoder_2.to(device, dtype=weight_dtype),
        "cloth_encoder": cloth_encoder.to(device, dtype=weight_dtype),
        "noise_scheduler": noise_scheduler,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2
    }
    
    pipeline = PromptDresser(
        vae=models["vae"],
        text_encoder=models["text_encoder"],
        text_encoder_2=models["text_encoder_2"],
        tokenizer=models["tokenizer"],
        tokenizer_2=models["tokenizer_2"],
        unet=models["unet"],
        scheduler=models["noise_scheduler"],
    ).to(device, dtype=weight_dtype)
    
    return {**models, "pipeline": pipeline}

models = load_models()
pipeline = models["pipeline"]

def generate_vton(person_image, cloth_image, outfit_prompt="", clothing_prompt=""):
    with tempfile.TemporaryDirectory() as tmp_dir:
        person_path = os.path.join(tmp_dir, "person.png")
        cloth_path = os.path.join(tmp_dir, "cloth.png")
        
        person_image.save(person_path)
        cloth_image.save(cloth_path)
        
        mask_path = os.path.join(tmp_dir, "mask.png")
        pose_path = os.path.join(tmp_dir, "pose.png")
        
        mask_image = generate_clothing_mask(person_path, label=4, output_path=mask_path, show_result=False)
        pose_image = generate_openpose(person_path, output_image_path=pose_path, show_result=False)
        
        auto_outfit_prompt = generate_caption(person_path, device)
        auto_clothing_prompt = generate_caption(cloth_path, device)
        
        final_outfit_prompt = outfit_prompt or auto_outfit_prompt
        final_clothing_prompt = clothing_prompt or auto_clothing_prompt
        
        with torch.autocast(device):
            result = pipeline(
                image=person_image,
                mask_image=mask_image,
                pose_image=pose_image,
                cloth_encoder=models["cloth_encoder"],
                cloth_encoder_image=cloth_image,
                prompt=final_outfit_prompt,
                prompt_clothing=final_clothing_prompt,
                height=1024,
                width=768,
                guidance_scale=2.0,
                guidance_scale_img=4.5,
                guidance_scale_text=7.5,
                num_inference_steps=30,
                strength=1,
                interm_cloth_start_ratio=0.5,
                generator=None,
            ).images[0]
    
    return result

with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container") as demo:
    gr.Markdown("# 🧥 Virtual Try-On")
    gr.Markdown("Загрузите фото человека и одежды для виртуальной примерки")
    
    with gr.Row():
        with gr.Column():
            person_input = gr.Image(label="Фото человека", type="pil", sources=["upload"])
            cloth_input = gr.Image(label="Фото одежды", type="pil", sources=["upload"])
            outfit_prompt = gr.Textbox(label="Описание образа (опционально)", placeholder="Например: man in casual outfit")
            clothing_prompt = gr.Textbox(label="Описание одежды (опционально)", placeholder="Например: red t-shirt with print")
            generate_btn = gr.Button("Сгенерировать примерку", variant="primary")
            
            gr.Examples(
                examples=[
                    ["./test/person2.png", "./test/00008_00.jpg", "man in skirt", "black longsleeve"]
                ],
                inputs=[person_input, cloth_input, outfit_prompt, clothing_prompt],
                label="Примеры для быстрого тестирования"
            )
            
        with gr.Column():
            output_image = gr.Image(label="Результат примерки", interactive=False)
    
    generate_btn.click(
        fn=generate_vton,
        inputs=[person_input, cloth_input, outfit_prompt, clothing_prompt],
        outputs=output_image
    )
    
    gr.Markdown("### Инструкция:")
    gr.Markdown("1. Загрузите четкое фото человека в полный рост\n"
                "2. Загрузите фото одежды на белом фоне\n"
                "3. При необходимости уточните описание образа или одежды\n"
                "4. Нажмите кнопку 'Сгенерировать примерку'")

if __name__ == "__main__":
    demo.queue(max_size=3).launch(
        server_name="0.0.0.0" if os.getenv("SPACE_ID") else None,
        share=os.getenv("GRADIO_SHARE") == "True",
        debug=True
    )