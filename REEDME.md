<h1 align="center">
    Blip2 Pretrain Code
    <br>
</h1>

<h4 align="center">
    <p>
        <b>简体中文</b>
    </p>
</h4>


blip2 预训练过程
# 项目描述
 此代码只是为了学习blip2预训练原理，最好debug看
 对于loss的具体解释可以看https://zhuanlan.zhihu.com/p/16034558568

# 代码组织解释
训练数据
data/images(图片) 和data/train_data.json(图片对应的解释)
model 文件夹是blip2的算法实现。此代码参考了https://github.com/salesforce/LAVIS/tree/ac8fc98c93c02e2dfb727e24a361c4c309c8dbbc
config.py配置文件，训练和test需要改动
load_data.py 加载训练数据
test和train 是测试和训练实现
output/model 保存训练好的模型

# 训练修改config.py 文件
visual_encoder_model_path: str = "eva_vit_g.pth" 下载地址：https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
qformer_model_path: str = "blip2_pretrained.pth"  下载地址：https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth
bert_base_uncased_path: str = "bert-base-uncased"  下载地址：https://huggingface.co/google-bert/bert-base-uncased
把以上三个模型地址改成你下载的地址


# 测试时修改配置文件
qformer_model_path: str = "blip2_pretrained.pth" 把这个地址改成你保存的地址

# 主要安装包版本
python=3.9.1
transformers==4.45.0
tokenizers==0.20.3
torch==2.5.1

## 贡献
欢迎通过提交拉取请求或在仓库中提出问题来为此模块做出贡献。

## License
本项目遵循[Apache-2.0开源协议](./LICENSE)