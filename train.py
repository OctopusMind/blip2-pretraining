from model import Blip2Qformer
from transformers import BlipImageProcessor
from torch.utils.data import DataLoader
from config import Config, ImageProcessorConfig, Blip2QformerConfig
from load_data import CustomDataset
from torch.optim import Adam
import torch


class TrainBlip2:
    def __init__(self):
        blip2_qformer_config = Blip2QformerConfig().__dict__
        image_processor_config = ImageProcessorConfig().__dict__
        self.config = Config()
        self.blip2model = Blip2Qformer(**blip2_qformer_config).to(self.config.device)  # 加载blip2
        self.processor = BlipImageProcessor(**image_processor_config)  # 加载图像预处理

        dataset = CustomDataset(self.config, self.processor, self.blip2model.tokenizer)  # 读取数据
        self.dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        self.model_opt = Adam(self.blip2model.parameters(), lr=self.config.lr)  # 设置优化器

    def train_blip2(self):
        for epochs in range(self.config.epochs):
            for i, data in enumerate(self.dataloader):
                loss = self.blip2model(data[0], data[1])
                self.blip2model.zero_grad()
                loss.loss.backward()
                self.model_opt.step()
                print(loss)
                # self.save_model()
            # 是否保存模型
            #self.save_model

    def save_model(self):
        blip2_pretrained = self.blip2model.state_dict()
        # 移除视觉编码器部分的权重
        blip2_pretrained = {k: v for k, v in blip2_pretrained.items() if not k.startswith("visual_encoder")}
        torch.save({"model":blip2_pretrained}, f"{self.config.save_model_path}/blip2_pretrained.pth")


if __name__ == '__main__':
    train_blip2 = TrainBlip2()
    train_blip2.train_blip2()
