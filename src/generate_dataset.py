import argparse
import csv
from pathlib import Path
import random
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from peft import PeftModel


def generate_synthetic_dataset(args):
    out_root = Path(args.output_dir)
    dir_normal = out_root / "NORMAL"
    dir_pneumonia = out_root / "PNEUMONIA"
    dir_normal.mkdir(parents=True, exist_ok=True)
    dir_pneumonia.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.csv"

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
        "NORMAL": "a frontal cxr_scan of normal lungs, clear lung fields, radiograph",
        "PNEUMONIA": "a frontal cxr_scan showing pneumonia, lung consolidation, radiograph",
    }
    negative_prompt = (
        "3d render, skeleton, glowing, neon, illustration, cartoon, drawing, "
        "artistic, transparent, perspective, bad anatomy, deformed"
    )

    targets = [
        ("NORMAL", dir_normal, args.num_normal, prompts["NORMAL"]),
        ("PNEUMONIA", dir_pneumonia, args.num_pneumonia, prompts["PNEUMONIA"]),
    ]

    manifest_exists = manifest_path.exists()
    
    with open(manifest_path, mode="a", newline="", encoding="utf-8") as f_manifest:
        writer = csv.writer(f_manifest)
        if not manifest_exists:
            writer.writerow(["filename", "relative_path", "label", "prompt", "negative_prompt", "seed", "guidance_scale", "steps"])

        for label, folder, total_count, prompt in targets:
            print(f"\nСтарт генерации класса {label}: {total_count} сэмплов...")
            
            pbar = tqdm(total=total_count, desc=f"Синтез {label}")
            generated = 0
            
            while generated < total_count:
                current_batch = min(args.batch_size, total_count - generated)
                
                # Генерация уникальных сидов для каждого сэмпла в батче
                seeds = [random.randint(0, 2**31 - 1) for _ in range(current_batch)]
                generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]
                
                # Батчевый инференс для ускорения генерации
                batch_prompts = [prompt] * current_batch
                batch_neg_prompts = [negative_prompt] * current_batch

                images = pipeline(
                    prompt=batch_prompts,
                    negative_prompt=batch_neg_prompts,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generators,
                ).images

                for idx, (img, seed) in enumerate(zip(images, seeds)):
                    current_idx = generated + idx + 1
                    file_name = f"synth_{label.lower()}_{current_idx:05d}.png"
                    save_path = folder / file_name
                    img.save(save_path)

                    rel_path = f"{label}/{file_name}"
                    writer.writerow([
                        file_name,
                        rel_path,
                        label,
                        prompt,
                        negative_prompt,
                        seed,
                        args.guidance_scale,
                        args.steps
                    ])

                generated += current_batch
                pbar.update(current_batch)
                f_manifest.flush()

            pbar.close()

    print(f"\nГенерация завершена. Датасет сохранен в: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Пакетная генерация синтетического датасета CXR")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora_dir", type=str, default="outputs/lora")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_dataset")
    parser.add_argument("--num_normal", type=int, default=4000, help="Количество снимков Normal (80%)")
    parser.add_argument("--num_pneumonia", type=int, default=1000, help="Количество снимков Pneumonia (20%)")
    parser.add_argument("--batch_size", type=int, default=4, help="Размер батча при инференсе")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    
    args = parser.parse_args()
    generate_synthetic_dataset(args)