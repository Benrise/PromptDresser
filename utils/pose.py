from controlnet_aux import OpenposeDetector
from PIL import Image
import torch


def generate_openpose(
    input_image_path: str,
    output_image_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_result: bool = False
) -> Image.Image:
    """
    Генерирует OpenPose карту позы из входного изображения.
    
    Параметры:
        input_image_path (str): Путь к исходному изображению
        output_image_path (str, optional): Путь для сохранения результата. Если None - не сохраняется.
        device (str): Устройство для обработки ('cuda' или 'cpu')
        show_result (bool): Показывать ли результат сразу
        
    Возвращает:
        Image.Image: Изображение с OpenPose картой позы
    """
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(device)
    
    image = Image.open(input_image_path).convert("RGB")
    
    openpose_map = openpose(image)
    
    if output_image_path:
        openpose_map.save(output_image_path)
    
    if show_result:
        openpose_map.show()
        
    return image
