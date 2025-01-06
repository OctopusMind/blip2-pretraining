import torch
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Config:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size: int = 5
    max_txt_len: int = 32
    epochs: int = 30
    lr: float = 0.001
    train_data_path: str = "data/train_data.json"
    images_path: str = "data/images"
    save_model_path: str = "output/model"  # 保存blip2模型地址


@dataclass
class Blip2QformerConfig:
    # Paths
    visual_encoder_model_path: str = "checkpoints/eva_vit_g.pth"
    qformer_model_path: str ="checkpoints、blip2_pretrained.pth"
    bert_base_uncased_path: str = "Downloads/bert-base-uncased"

    # ViT encoder
    img_size: int = 224
    drop_path_rate: float = 0.0
    freeze_vit: bool = True

    # Q-Former
    num_query_token: int = 32


@dataclass
class ImageProcessorConfig:
    do_convert_rgb: bool = True
    do_normalize: bool = True
    do_rescale: bool = True
    do_resize: bool = True
    image_mean: List[float] = field(default_factory=lambda: [0.48145466, 0.4578275, 0.40821073])
    image_std: List[float] = field(default_factory=lambda: [0.26862954, 0.26130258, 0.27577711])
    resample: int = 3
    rescale_factor: float = 0.00392156862745098
    size: Dict[str, int] = field(default_factory=lambda: {"height": 224, "width": 224})
