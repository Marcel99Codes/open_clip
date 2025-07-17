from dataclasses import dataclass
from typing import Optional

@dataclass
class TransformerArgs:
    tokenization_pipeline: Optional[str] = None
    grayscale_only: Optional[bool] = None
    conv1_patch_size: Optional[int] = None
    convb_patch_size: Optional[int] = None

_transformer_args = TransformerArgs()

def set_transformer_args(tokenization_pipeline: str,
                         conv1_patch_size: int,
                         convb_patch_size: int,
                         grayscale_only: bool) -> None:
    _transformer_args.tokenization_pipeline = tokenization_pipeline
    _transformer_args.conv1_patch_size = conv1_patch_size
    _transformer_args.convb_patch_size = convb_patch_size
    _transformer_args.grayscale_only = grayscale_only


def get_tokenization_pipeline() -> Optional[str]:
    return _transformer_args.tokenization_pipeline


def get_grayscale_only() -> Optional[bool]:
    return _transformer_args.grayscale_only

def get_conv1_patch_size() -> Optional[int]:
    return _transformer_args.conv1_patch_size


def get_convb_patch_size() -> Optional[int]:
    return _transformer_args.convb_patch_size
