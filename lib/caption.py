from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


def generate_caption(image_path, device="cuda"):
    print("Генерация подписи...")
    processor = AutoProcessor.from_pretrained("microsoft/git-base", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(device)
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        pixel_values=inputs.pixel_values, 
        max_length=50,
        pad_token_id=processor.tokenizer.pad_token_id
    )
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Сгенерированная подпись:", caption)
    return caption