import argparse
import os
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(
        description="Генерация рентгеновских снимков с весами LoRA"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Базовая модель",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default="outputs/lora",
        help="Путь к папке с обученными весами LoRA",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/lora_samples",
        help="Папка для сохранения сгенерированных снимков",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Количество сэмплов на каждый класс",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Количество шагов денойзинга",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="Шкала соответствия промпту (CFG scale)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Базовый сид для генератора случайных чисел",
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Загрузка базовой модели: {args.model_id}...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    print(f"Подключение весов LoRA через PEFT из: {args.lora_dir}...")
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args.lora_dir)

    prompts = {
        "normal": "a frontal cxr_scan of normal lungs, clear lung fields, radiograph",
        "pneumonia": "a frontal cxr_scan showing pneumonia, lung consolidation, radiograph",
    }
    negative_prompt = (
            "3d render, skeleton, glowing, neon, illustration, cartoon, drawing, "
            "artistic, transparent, perspective, bad anatomy, deformed"
        )
    print("Старт генерации снимков...")
    for class_name, prompt in prompts.items():
        print(f"\nГенерация сэмплов для класса '{class_name}'...")
        for i in range(args.samples):
            generator = torch.Generator(device=device).manual_seed(
                args.seed + i
            )
            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

            save_file = out_path / f"lora_{class_name}_sample_{i+1}.png"
            image.save(save_file)
            print(f"Сохранено: {save_file}")

    print(
        f"\nГенерация успешно завершена! Снимки сохранены в: {args.output_dir}"
    )


if __name__ == "__main__":
    main()