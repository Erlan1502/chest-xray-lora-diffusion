import argparse
import json
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm


def prepare_dataset(raw_dir: Path, output_dir: Path, img_size: int = 512):
    """
    1. Ищет все снимки в train/NORMAL и train/PNEUMONIA.
    2. Приводит к RGB и масштабирует до img_size x img_size.
    3. Сохраняет в processed/images/ и генерирует metadata.jsonl для обучения диффузии.
    """
    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    # Классы и соответствующие текстовые промпты
    prompts = {
        "NORMAL": "a frontal cxr_scan of normal lungs, clear lung fields, radiograph",
        "PNEUMONIA": "a frontal cxr_scan showing pneumonia, lung consolidation, radiograph",
    }

    metadata_records = []
    supported_extensions = {".jpeg", ".jpg", ".png"}

    print(f"Поиск изображений в: {raw_dir}")

    for class_name, prompt_text in prompts.items():
        class_folder = raw_dir / "train" / class_name

        if not class_folder.exists():
            print(f"Внимание: папка {class_folder} не найдена!")
            continue

        image_files = [
            f for f in class_folder.iterdir()
            if f.suffix.lower() in supported_extensions
        ]

        print(f"Обработка класса '{class_name}': найдено {len(image_files)} файлов...")

        for file_path in tqdm(image_files, desc=f"Класс {class_name}"):
            try:
                # Открываем и конвертируем в трехканальный RGB (для VAE Stable Diffusion)
                with Image.open(file_path) as img:
                    img = img.convert("RGB")
                    # Качественный ресайз с билинейным/ланцош фильтром
                    img_resized = ImageOps.pad(
                        img, 
                        (img_size, img_size), 
                        method=Image.Resampling.LANCZOS, 
                        color=(0, 0, 0), 
                        centering=(0.5, 0.5)
                    )
                    new_filename = f"{class_name.lower()}_{file_path.name}"
                    save_path = images_out_dir / new_filename
                    img_resized.save(save_path, format="JPEG", quality=95)

                    # Запись для Hugging Face datasets: file_name относительно папки с metadata.jsonl
                    metadata_records.append({
                        "file_name": f"images/{new_filename}",
                        "text": prompt_text,
                        "label": class_name
                    })
            except Exception as e:
                print(f"Пропущен поврежденный файл {file_path.name}: {e}")

    # Запись metadata.jsonl
    with open(metadata_path, "w", encoding="utf-8") as f:
        for record in metadata_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nПодготовка завершена успешно!")
    print(f"Всего сохранено снимков: {len(metadata_records)}")
    print(f"Файл метаданных: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подготовка датасета Chest X-ray для обучения LoRA")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw/chest_xray"),
        help="Путь к исходной распакованной папке chest_xray"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Путь для сохранения нормализованных данных"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Разрешение изображения (по умолчанию 512x512)"
    )

    args = parser.parse_args()
    prepare_dataset(args.raw_dir, args.output_dir, args.size)