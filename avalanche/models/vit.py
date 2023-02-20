from avalanche.models import SimpleMLP


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    from avalanche.models.timm_vit import resolve_pretrained_cfg, \
                build_model_with_cfg, VisionTransformer, checkpoint_filter_fn

    if kwargs.get("features_only", None):
        raise RuntimeError("features_only not implemented for \
                            Vision Transformer models.")

    pretrained_cfg = resolve_pretrained_cfg(
        variant, pretrained_cfg=kwargs.pop("pretrained_cfg", None)
    )
    model = build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in pretrained_cfg["url"],
        **kwargs,
    )
    return model


def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ViT-Tiny (Vit-Ti/16)"""
    model_kwargs = dict(
                        patch_size=16, embed_dim=192,
                        depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "vit_tiny_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_tiny_patch16_384(pretrained=False, **kwargs):
    """ViT-Tiny (Vit-Ti/16) @ 384x384."""
    model_kwargs = dict(
                        patch_size=16, embed_dim=192,
                        depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "vit_tiny_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch32_224(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/32)"""
    model_kwargs = dict(
                        patch_size=32, embed_dim=384,
                        depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch32_384(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/32) at 384x384."""
    model_kwargs = dict(
                        patch_size=32, embed_dim=384,
                        depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch16_224(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights 
    with the small variant from the DeiT paper
    """
    model_kwargs = dict(
                        patch_size=16, embed_dim=384,
                        depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch16_384(pretrained=False, **kwargs):
    """ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights 
    with the small variant from the DeiT paper
    """
    model_kwargs = dict(
                        patch_size=16, embed_dim=384,
                        depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch32_224(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, 
    source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
                        patch_size=32, embed_dim=768,
                        depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch32_384(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, 
    source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
                        patch_size=32, embed_dim=768,
                        depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, 
    source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
                        patch_size=16, embed_dim=768,
                        depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_384(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, 
    source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
                        patch_size=16, embed_dim=768,
                        depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch8_224(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, 
    source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
                        patch_size=8, embed_dim=768,
                        depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch32_224(pretrained=False, **kwargs):
    """
    ViT-Large (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929)
    No pretrained weights."""
    model_kwargs = dict(
                        patch_size=32, embed_dim=1024,
                        depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch32_384(pretrained=False, **kwargs):
    """
    ViT-Large (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, 
    source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
                        patch_size=32, embed_dim=1024,
                        depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Large (ViT-L/16) from original paper 
    (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, 
    source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
                        patch_size=16, embed_dim=1024,
                        depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch16_384(pretrained=False, **kwargs):
    """
    ViT-Large (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, 
    source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
                        patch_size=16, embed_dim=1024,
                        depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch14_224(pretrained=False, **kwargs):
    """ViT-Large model (ViT-L/14)"""
    model_kwargs = dict(
                        patch_size=14, embed_dim=1024,
                        depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch14_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_huge_patch14_224(pretrained=False, **kwargs):
    """
    ViT-Huge (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929)
    """
    model_kwargs = dict(
                        patch_size=14, embed_dim=1280,
                        depth=32, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_huge_patch14_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_giant_patch14_224(pretrained=False, **kwargs):
    """
    ViT-Giant (ViT-g/14) from `Scaling Vision Transformers`:
    https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(
        patch_size=14,
        embed_dim=1408,
        mlp_ratio=48 / 11,
        depth=40,
        num_heads=16,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_giant_patch14_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_gigantic_patch14_224(pretrained=False, **kwargs):
    """
    ViT-Gigantic model (ViT-G/14) from `Scaling Vision Transformers`:
    https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(
        patch_size=14,
        embed_dim=1664,
        mlp_ratio=64 / 13,
        depth=48,
        num_heads=16,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_gigantic_patch14_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_tiny_patch16_224_in21k(pretrained=False, **kwargs):
    """
    ViT-Tiny (Vit-Ti/16).
    ImageNet-21k weights @ 224x224, 
    source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and 
    no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, 
                        depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "vit_tiny_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch32_224_in21k(pretrained=False, **kwargs):
    """
    ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, 
    source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and 
    no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, 
                        depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch16_224_in21k(pretrained=False, **kwargs):
    """
    ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, 
    source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and 
    no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, 
                        depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, 
    source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and 
    no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, 
                        depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, 
    source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and 
    no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, 
                        depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch8_224_in21k(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, 
    source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and 
    no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, 
                        depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    """
    ViT-Large (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, 
    source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer 
    but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, 
                        depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    """
    ViT-Large (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, 
    source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and 
    no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, 
                        depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """
    ViT-Huge (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, 
    source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer 
    but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(patch_size=14, embed_dim=1280, 
                        depth=32, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_huge_patch14_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_224_sam(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/16) w/ SAM pretrained weights. 
    Paper: https://arxiv.org/abs/2106.01548
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, 
                        depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224_sam", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch32_224_sam(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/32) w/ SAM pretrained weights. 
    Paper: https://arxiv.org/abs/2106.01548
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, 
                        depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_224_sam", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch16_224_dino(pretrained=False, **kwargs):
    """
    ViT-Small (ViT-S/16) w/ DINO pretrained weights (no head)
    https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, 
                        depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch16_224_dino", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch8_224_dino(pretrained=False, **kwargs):
    """
    ViT-Small (ViT-S/8) w/ DINO pretrained weights (no head)
    https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=8, embed_dim=384, 
                        depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch8_224_dino", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_224_dino(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/16) /w DINO pretrained weights (no head)
    https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, 
                        depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224_dino", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch8_224_dino(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/8) w/ DINO pretrained weights (no head)
    https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=8, embed_dim=768,
                        depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_224_dino", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_224_miil_in21k(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, 
        num_heads=12, qkv_bias=False, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch16_224_miil_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_224_miil(pretrained=False, **kwargs):
    """
    ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, 
        num_heads=12, qkv_bias=False, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch16_224_miil", pretrained=pretrained, **model_kwargs
    )
    return model


# Experimental models below


def vit_base_patch32_plus_256(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/32+)"""
    model_kwargs = dict(
        patch_size=32, embed_dim=896, depth=12, 
        num_heads=14, init_values=1e-5, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch32_plus_256", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_plus_240(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16+)"""
    model_kwargs = dict(
        patch_size=16, embed_dim=896, depth=12, 
        num_heads=14, init_values=1e-5, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_patch16_plus_240", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch16_36x1_224(pretrained=False, **kwargs):
    """
    ViT-Base w/ LayerScale + 36 x 1 (36 block serial) config. 
    Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers`
    https://arxiv.org/abs/2203.09795
    Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=36, 
        num_heads=6, init_values=1e-5, **kwargs
    )
    model = _create_vision_transformer(
        "vit_small_patch16_36x1_224", pretrained=pretrained, **model_kwargs
    )
    return model


# Only for unittest
def simpleMLP(num_classes=10, **kwargs):
    model = SimpleMLP(input_size=6, hidden_size=10, 
                      num_classes=num_classes)
    return model


def create_model(model_name='', **kwargs):
    get_model = globals()[model_name]
    return get_model(**kwargs)
