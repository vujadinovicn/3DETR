#!/bin/bash

SCENE_IDXS=(16 39 56 57 58 67 75 144)

for idx in "${SCENE_IDXS[@]}"; do
    python demo.py \
        --scene_idx "$idx" \
        --use_3d_nms True \
        --test_ckpt scannet_ep1080.pth \
        --dump_dir "last_qualitative/" \
        --dataset_name scannet
done
