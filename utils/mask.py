from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import numpy as np
import requests
import torch.nn.functional as F
import torch
import os

def generate_clothing_mask(
    image_path: str,
    label: int,
    output_path: str = "./output_mask.png",
    model_name: str = "mattmdjaga/segformer_b2_clothes",
    show_result: bool = False
) -> Image.Image:
    """
    Генерирует бинарную маску для указанного класса одежды и сохраняет её
    
    Args:
        image_path: Путь к изображению или URL
        label: Класс для сегментации (0-17)
        output_path: Путь для сохранения маски
        model_name: Название модели HuggingFace
        show_result: Показать результат matplotlib
        
    Returns:
        PIL.Image: Бинарная маска (белый - выбранный класс, черный - остальное)
    """

    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    
    if image_path.startswith(('http://', 'https://')):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_np = np.array(image)
    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
        raise ValueError("Изображение должно быть в формате RGB (H, W, 3)")
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    upsampled_logits = F.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    mask = (pred_seg == label).numpy().astype('uint8') * 255
    mask_image = Image.fromarray(mask)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mask_image.save(output_path)
    
    if show_result:
        import matplotlib.pyplot as plt
        plt.imshow(mask_image, cmap='gray')
        plt.title(f"Mask for label {label}")
        plt.axis('off')
        plt.show()
    
    return mask_image
