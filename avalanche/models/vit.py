from avalanche.models import SimpleMLP


def vit_tiny_patch16_224(pretrained=True, **kwargs):
    """ViT-Tiny (Vit-Ti/16)"""
    from avalanche.models.timm_vit import _create_vision_transformer

    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "vit_tiny_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch16_224(pretrained=True, **kwargs):
    """ViT-Base (ViT-B/16) from original paper
    (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224,
    source https://github.com/google-research/vision_transformer.
    """
    from avalanche.models.timm_vit import _create_vision_transformer

    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Large (ViT-L/16) from original paper
    (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224,
    source https://github.com/google-research/vision_transformer.
    """
    from avalanche.models.timm_vit import _create_vision_transformer

    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_small_patch32_224(pretrained=True, **kwargs):
    """ViT-Small (ViT-S/32)"""
    from avalanche.models.timm_vit import _create_vision_transformer

    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_small_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_base_patch32_224(pretrained=True, **kwargs):
    """
    ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k,
    source https://github.com/google-research/vision_transformer.
    """
    from avalanche.models.timm_vit import _create_vision_transformer

    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


def vit_large_patch32_224(pretrained=True, **kwargs):
    """
    ViT-Large (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929)
    No pretrained weights."""
    from avalanche.models.timm_vit import _create_vision_transformer

    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


# Only for unittest
def simpleMLP(num_classes=10, **kwargs):
    model = SimpleMLP(input_size=6, hidden_size=10, num_classes=num_classes)
    return model


def create_model(model_name="", **kwargs):
    get_model = globals()[model_name]
    return get_model(**kwargs)
