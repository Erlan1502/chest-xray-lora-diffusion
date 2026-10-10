import argparse
import json
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import random
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model


class ChestXRayDataset(Dataset):
    def __init__(self, metadata_path: Path, tokenizer, max_length: int = 77):
        self.root_dir = metadata_path.parent
        self.entries = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                self.entries.append(json.loads(line.strip()))

        self.tokenizer = tokenizer
        self.max_length = max_length
        # Стандартная нормализация в диапазон [-1, 1] для входа в VAE
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        image_path = self.root_dir / item["file_name"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_transforms(image)
        prompt_text = "" if random.random() < 0.1 else item["text"]
        inputs = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids.squeeze(0)
        }


def train_lora(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Внимание: запуск без CUDA! Для обучения требуется GPU с поддержкой CUDA.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка компонентов базовой модели: {args.model_id}...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    # 1. Замораживаем VAE и CLIP Text Encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 2. Включаем gradient checkpointing для жесткой экономии VRAM
    unet.enable_gradient_checkpointing()

    # 3. Конфигурация PEFT / LoRA для слоев Cross-Attention
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Датасет и DataLoader
    dataset = ChestXRayDataset(Path(args.metadata_path), tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.lr,
        weight_decay=1e-2
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    print("\nСтарт тренировочного цикла LoRA...")

    global_step = 0
    unet.train()

    for epoch in range(args.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                # Кодируем изображения в латентное пространство (латенты 64x64)
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Добавляем гауссовский шум
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                # UNet с LoRA предсказывает остаточный шум
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Сохраняем адаптер LoRA
    print(f"\nСохранение обученных весов LoRA в {output_dir}...")
    unet.save_pretrained(output_dir)
    print("Обучение успешно завершено!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение LoRA для Stable Diffusion 1.5 на рентгенограммах")
    parser.add_argument("--metadata_path", type=str, default="data/processed/metadata.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/lora")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)

    args = parser.parse_args()
    train_lora(args)