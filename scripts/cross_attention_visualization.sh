#!/bin/bash

SCENE_IDXS=(197 198)
# SCENE_IDXS=(0 60 67 77)

for idx in "${SCENE_IDXS[@]}"; do
    python visualize_attention.py \
        --scene_idx "$idx" \
        --test_ckpt scannet_ep1080.pth \
        --output_dir "final_attention_2/" \
        --dataset_name scannet
done
