# Copyright (c) Facebook, Inc. and its affiliates.
from .model_3detr import build_3detr
from .model_3detr_cls import build_3detr_cls

MODEL_FUNCS = {
    "3detr": build_3detr,
    "3detr_cls": build_3detr_cls,
}

def build_model(args, dataset_config):
    model, processor = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model, processor