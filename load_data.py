from PIL import Image
from torch.utils.data import Dataset
import json


class CustomDataset(Dataset):
    def __init__(self, config, processor, tokenizer):
        self.config = config
        with open(self.config.train_data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one = self.data[idx]
        image_path = f"{self.config.images_path}/{one['image_id']}"
        text = one["rationales"][one["correct_choice_idx"] - 1]

        # 加载图片
        image = Image.open(image_path)
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_txt_len,
            return_tensors="pt",
        )#.to(self.config.device)
        for key, val in text_tokens.items():
            text_tokens[key] = val.squeeze(0)

        # 处理输入数据
        inputs = self.processor(images=image, return_tensors="pt", padding=True)["pixel_values"].squeeze(0)#.to(self.config.device)
        return inputs, text_tokens
