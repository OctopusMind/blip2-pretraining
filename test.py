from model import Blip2Qformer
from transformers import BlipImageProcessor
from config import ImageProcessorConfig, Blip2QformerConfig
from PIL import Image


class TestBlip2:
    def __init__(self):
        blip2_qformer_config = Blip2QformerConfig().__dict__
        image_processor_config = ImageProcessorConfig().__dict__
        self.blip2model = Blip2Qformer(**blip2_qformer_config).eval()  # 加载blip2
        self.processor = BlipImageProcessor(**image_processor_config)  # 加载图像预处理

    def test_blip2(self, one_images_path):
        image = Image.open(one_images_path)
        inputs = self.processor(images=image, return_tensors="pt", padding=True)["pixel_values"]
        output = self.blip2model.generate(inputs)[0]
        print(f"{one_images_path} 图像的描述是：{output}")
        return output


if __name__ == '__main__':
    test_blip2 = TestBlip2()
    res = test_blip2.test_blip2("data/images/0.png")
    print(res)
