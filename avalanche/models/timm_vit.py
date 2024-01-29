""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision 
Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at: 
https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's 
https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's: 
https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and 
Tensorflow Bert
Hacked together by / Copyright 2020, Ross Wightman
# ------------------------------------------
# Based on code added by:
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from avalanche.models import Prompt

try:
    from timm.models.helpers import (
        adapt_input_conv,
        checkpoint_seq,
        resolve_pretrained_cfg,
        build_model_with_cfg,
    )
    from timm.models.layers import PatchEmbed
    from timm.models.vision_transformer import VisionTransformer, Block

except ImportError:
    raise ModuleNotFoundError(
        "timm is not found, "
        "if you want to use L2P "
        "please install avalanche with the "
        "detection dependencies: "
        "pip install avalanche-lib[l2p]"
    )


class ViTWithPrompt(VisionTransformer):
    """
    Visual Transformer with Prompt. This class add prompts to a visual
    transformer to implement the Method Learning to Prompt (L2P)

    Implementation based on VisionTransformer from timm library
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=None,
        class_token=True,
        no_embed_class=False,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
        prompt_length=None,
        embedding_key="cls",
        prompt_init="uniform",
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init="uniform",
        head_type="token",
        use_prompt_mask=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
            block_fn: (nn.Module): transformer block
            prompt_pool (bool): use prompt pool or not
        """
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            class_token=class_token,
            no_embed_class=no_embed_class,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
        )

        self.class_token = class_token
        num_patches = self.patch_embed.num_patches

        if no_embed_class:
            embed_len = num_patches
        else:
            embed_len = num_patches + self.num_prefix_tokens

        if prompt_length is not None and pool_size is not None and prompt_pool:
            embed_len += prompt_length * top_k
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.prompt_pool = prompt_pool
        self.head_type = head_type
        self.use_prompt_mask = use_prompt_mask

        if prompt_length is not None and pool_size is not None and prompt_pool:
            self.prompt = Prompt(
                length=prompt_length,
                embed_dim=embed_dim,
                embedding_key=embedding_key,
                prompt_init=prompt_init,
                prompt_pool=prompt_pool,
                prompt_key=prompt_key,
                pool_size=pool_size,
                top_k=top_k,
                batchwise_prompt=batchwise_prompt,
                prompt_key_init=prompt_key_init,
            )

        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        if weight_init != "skip":
            self.init_weights(weight_init)

    def forward_features(self, x, task_id=-1, cls_features=None, train=False):
        x = x.float()
        x = self.patch_embed(x)

        if hasattr(self, "prompt"):
            if self.use_prompt_mask and train:
                start = task_id * self.prompt.top_k
                end = (task_id + 1) * self.prompt.top_k
                single_prompt_mask = torch.arange(start, end).to(x.device)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                if end > self.prompt.pool_size:
                    prompt_mask = None
            else:
                prompt_mask = None
            res = self.prompt(x, prompt_mask=prompt_mask, cls_features=cls_features)
            self.total_prompt_len = res["total_prompt_len"]
            x = res["prompted_embedding"]
        else:
            res = dict()
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        res["x"] = x

        return res

    def forward_head(self, res, pre_logits: bool = False):
        x = res["x"]
        if self.class_token and self.head_type == "token":
            x = x[:, 0]
        elif self.head_type == "gap" and self.global_pool == "avg":
            x = x.mean(dim=1)
        elif self.head_type == "prompt" and self.prompt_pool:
            x = (
                x[:, 1 : (1 + self.total_prompt_len)]
                if self.class_token
                else x[:, 0 : self.total_prompt_len]
            )
            x = x.mean(dim=1)
        elif self.head_type == "token+prompt" and self.prompt_pool and self.class_token:
            x = x[:, 0 : self.total_prompt_len + 1]
            x = x.mean(dim=1)
        else:
            raise ValueError(f"Invalid classifier={self.classifier}")

        res["pre_logits"] = x

        x = self.fc_norm(x)

        res["logits"] = self.head(x)

        return res

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        res = self.forward_features(
            x, task_id=task_id, cls_features=cls_features, train=train
        )
        res = self.forward_head(res)
        return res

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ""):
    """
    Load weights from .npz checkpoints for official
    Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and "opt/target/embedding/kernel" in w:
        prefix = "opt/target/"

    if hasattr(model.patch_embed, "backbone"):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, "stem")
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(
            adapt_input_conv(
                stem.conv.weight.shape[1], _n2p(w[f"{prefix}conv_root/kernel"])
            )
        )
        stem.norm.weight.copy_(_n2p(w[f"{prefix}gn_root/scale"]))
        stem.norm.bias.copy_(_n2p(w[f"{prefix}gn_root/bias"]))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f"{prefix}block{i + 1}/unit{j + 1}/"
                    for r in range(3):
                        getattr(block, f"conv{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}conv{r + 1}/kernel"])
                        )
                        getattr(block, f"norm{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/scale"])
                        )
                        getattr(block, f"norm{r + 1}").bias.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/bias"])
                        )
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(
                            _n2p(w[f"{bp}conv_proj/kernel"])
                        )
                        block.downsample.norm.weight.copy_(
                            _n2p(w[f"{bp}gn_proj/scale"])
                        )
                        block.downsample.norm.bias.copy_(_n2p(w[f"{bp}gn_proj/bias"]))
        embed_conv_w = _n2p(w[f"{prefix}embedding/kernel"])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f"{prefix}embedding/kernel"])
        )
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
    model.cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
    pos_embed_w = _n2p(w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = (
            # resize pos embedding when different size from pretrained weights
            resize_pos_embed(
                pos_embed_w,
                model.pos_embed,
                getattr(model, "num_prefix_tokens", 1),
                model.patch_embed.grid_size,
            )
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
    model.norm.bias.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))
    if (
        isinstance(model.head, nn.Linear)
        and model.head.bias.shape[0] == w[f"{prefix}head/bias"].shape[-1]
    ):
        model.head.weight.copy_(_n2p(w[f"{prefix}head/kernel"]))
        model.head.bias.copy_(_n2p(w[f"{prefix}head/bias"]))
    # NOTE representation layer has been removed,
    # not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear)
    #               and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.\
    #               copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.\
    #               copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
        mha_prefix = block_prefix + "MultiHeadDotProductAttention_1/"
        block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"]))
        block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"]))
        block.attn.qkv.weight.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T
                    for n in ("query", "key", "value")
                ]
            )
        )
        block.attn.qkv.bias.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1)
                    for n in ("query", "key", "value")
                ]
            )
        )
        block.attn.proj.weight.copy_(_n2p(w[f"{mha_prefix}out/kernel"]).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"]))
        for r in range(2):
            getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"])
            )
            getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"])
            )
        block.norm2.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/scale"]))
        block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/bias"]))


def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict.
    # Adapted from:
    # https://github.com/google-research/vision_transformer/blob/
    #       00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    # modify
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = (
            posemb[:, :num_prefix_tokens],
            posemb[0, num_prefix_tokens:],
        )
        # ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if ntok_new > gs_old**2:
        ntok_new -= gs_old**2
        # expand cls's pos embedding for prompt tokens
        posemb_prefix = posemb_prefix.expand(-1, ntok_new, -1)
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode="bicubic", align_corners=False
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model, adapt_layer_scale=False):
    """
    convert patch embedding weight from manual patchify + linear proj to conv
    """

    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]

    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, _, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == "pos_embed" and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size
            # from pretrained weights
            v = resize_pos_embed(
                v,
                model.pos_embed,
                (
                    0
                    if getattr(model, "no_embed_class")
                    else getattr(model, "num_prefix_tokens", 1)
                ),
                model.patch_embed.grid_size,
            )
        elif adapt_layer_scale and "gamma_" in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r"gamma_([0-9])", r"ls\1.gamma", k)
        elif "pre_logits" in k:
            # NOTE representation layer removed as not used in latest 21k/1k
            # pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for \
                            Vision Transformer models."
        )

    pretrained_cfg = resolve_pretrained_cfg(
        variant, pretrained_cfg=kwargs.pop("pretrained_cfg", None)
    )
    model = build_model_with_cfg(
        ViTWithPrompt,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in pretrained_cfg["url"],
        **kwargs,
    )
    return model
