import torch
import torch.nn as nn
from transformers import BertTokenizer
from .Qformer import BertConfig, BertLMHeadModel
from .standard_output import BlipOutput, LayerNorm
from torch.nn import functional as F
from .visiontransformer import VisionTransformer
from functools import partial


class Blip2Qformer(nn.Module):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    """

    def __init__(
            self,
            img_size=224,
            drop_path_rate=0,
            freeze_vit=True,
            num_query_token=32,
            cross_attention_freq=2,
            embed_dim=256,
            max_txt_len=32,
            visual_encoder_model_path=None,
            qformer_model_path=None,
            bert_base_uncased_path=None
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer(bert_base_uncased_path)

        self.visual_encoder = self.init_vision_encoder(img_size, drop_path_rate)
        self.ln_vision = LayerNorm(self.visual_encoder.num_features)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
        self.Qformer, self.query_tokens = self.init_qformer(
            num_query_token, self.visual_encoder.num_features, bert_base_uncased_path, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.load_model_weights(visual_encoder_model_path, qformer_model_path)

    def init_vision_encoder(self, img_size, drop_path_rate):
        # 此处加载的是eva_clip_g模型， 加载需要改一下
        visual_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=14,
            use_mean_pooling=False,
            embed_dim=1408,
            depth=39,
            num_heads=1408 // 88,
            mlp_ratio=4.3637,
            qkv_bias=True,
            drop_path_rate=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_checkpoint=False,
        )
        return visual_encoder

    def load_model_weights(self, vision_weights_path, qformer_model_path):
        # 加载视觉编码器权重
        vision_weights = torch.load(vision_weights_path, map_location="cpu")
        self.visual_encoder.load_state_dict(vision_weights, strict=False)
        print("Visual encoder weights loaded successfully!")

        # 加载其他权重
        rest_weights = torch.load(qformer_model_path, map_location="cpu")["model"]
        model_dict = self.state_dict()
        # 过滤掉视觉编码器的权重
        rest_weights = {k: v for k, v in rest_weights.items() if not k.startswith("visual_encoder")}
        model_dict.update(rest_weights)
        self.load_state_dict(model_dict)
        print("Remaining model weights loaded successfully!")

    def init_qformer(self, num_query_token, vision_width, bert_base_uncased_path, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(bert_base_uncased_path)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            bert_base_uncased_path, config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def init_tokenizer(self, bert_base_uncased_path, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained(bert_base_uncased_path,
                                                  truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def forward(self, image, text_tokens):
        image_embeds = self.ln_vision(self.visual_encoder(image))  # 视觉encode, ln_vision是标准化
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # query_tokens 是原理图中的learned queries

        # 获得queries和图像融合的encode
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        # 获得文本encode
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        # 下面的loss 计算可以看博客https://zhuanlan.zhihu.com/p/16034558568
        ###============== Image-text Contrastive ===================###
        image_feats_all = image_feats  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = text_feat  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        # rank = dist.get_rank()
        rank = 0
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=torch.long).to(
            image.device
        )

        loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) + F.cross_entropy(sim_t2i, targets,
                                                                                             label_smoothing=0.1)) / 2

        ###============== Image-text Matching ===================###
        text_input_ids_world = text_tokens.input_ids
        text_attention_mask_world = text_tokens.attention_mask
        image_embeds_world = image_embeds  # all_gather_with_grad(image_embeds)
        with torch.no_grad():

            sim_t2i[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)
            sim_i2t[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)

            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss
        return BlipOutput(
            loss=(loss_itc + loss_itm + loss_lm).float() if isinstance(loss_itc, torch.Tensor) else None,
            loss_itc=loss_itc.float() if isinstance(loss_itc, torch.Tensor) else None,
            loss_itm=loss_itm.float() if isinstance(loss_itm, torch.Tensor) else None,
            loss_lm=loss_lm.float() if isinstance(loss_lm, torch.Tensor) else None,
        )

    @torch.no_grad()
    def generate(
            self,
            image,
            use_nucleus_sampling=False,
            num_beams=1,
            max_length=30,
            min_length=10,
            top_p=0.9,
            repetition_penalty=1.0,
    ):
        """
        Args:
            image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions
