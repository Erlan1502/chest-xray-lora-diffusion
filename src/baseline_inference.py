import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline


def run_baseline_inference(output_dir: Path, num_samples_per_prompt: int = 3):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Запуск Baseline-инференса...")
    print(f"Используемое устройство: {device.upper()} | Тип данных: {torch_dtype}")

    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"Загрузка базовой предобученной модели {model_id}...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None  # Отключаем дефолтный цензор для ускорения и исключения ложных срабатываний
    )
    pipe = pipe.to(device)

    # Опорные промпты (совпадают с разметкой датасета)
    test_prompts = [
        ("normal", "a frontal chest x-ray of normal lungs, clear lung fields, radiograph"),
        ("pneumonia", "a frontal chest x-ray showing pneumonia, lung consolidation, radiograph")
    ]

    generator = torch.Generator(device=device).manual_seed(42)

    for label, prompt in test_prompts:
        print(f"\nГенерация сэмплов для класса '{label}'...")
        for i in range(num_samples_per_prompt):
            image = pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator
            ).images[0]

            filename = f"baseline_{label}_sample_{i+1}.png"
            save_path = output_dir / filename
            image.save(save_path)
            print(f"Сохранено: {save_path}")

    print("\nBaseline-инференс успешно завершен! Сэмплы сохранены в outputs/baseline/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск базовой генерации SD 1.5 без дообучения")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/baseline"),
        help="Папка для сохранения сгенерированных контрольных снимков"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Количество изображений на каждый промпт"
    )
    args = parser.parse_args()
    run_baseline_inference(args.output_dir, args.samples)