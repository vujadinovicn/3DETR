# 3DETR: Review, Reproduction and Extensions

Course project (MVA, NPM3D) built on the official [3DETR](https://github.com/facebookresearch/3detr) code (Misra et al., ICCV 2021). We reproduce a subset of the paper's claims on ScanNetV2, re-implement the analyses the official repo does not ship (query-budget and NMS ablations, ModelNet40 classification with the 3DETR encoder), and add two analyses beyond the paper: recall as a function of object size / point count, and decoder cross-attention visualization.

**Report:** [docs/report.pdf](docs/report.pdf)

---

## Table of Contents

- [Summary of Contributions](#summary-of-contributions)
- [Key Findings](#key-findings)
- [What Changed vs. Upstream](#what-changed-vs-upstream)
- [How to Run Our Additions](#how-to-run-our-additions)
- [Original 3DETR README](#original-3detr-readme) (installation, datasets, pretrained models, training, citation)

---

## Summary of Contributions

| Contribution | Where |
|---|---|
| Reproduction of 3DETR / 3DETR-m on ScanNetV2 with the official 1080-epoch checkpoints | [inference.py](inference.py), stock `main.py --test_only` |
| Number-of-queries sweep: AP and inference time, 3DETR vs VoteNet | [experiments/query_effect_analysis.py](experiments/query_effect_analysis.py) |
| NMS ablation: none / 2D / 3D class-agnostic / 3D per-class, from a single set of predictions | [experiments/nms_effect_analysis.py](experiments/nms_effect_analysis.py) |
| Encoder-only classification on ModelNet40 (CLS token for 3DETR, max-pool for 3DETR-m) | [models/model_3detr_cls.py](models/model_3detr_cls.py), [datasets/modelnet40.py](datasets/modelnet40.py), `*_cls` functions in [engine.py](engine.py) and [main.py](main.py) |
| Recall by object volume, point count and point density (not in the paper) | [experiments/binned_recall_analysis.py](experiments/binned_recall_analysis.py), [utils/ap_calculator.py](utils/ap_calculator.py) |
| Query-to-point cross-attention maps from the decoder (not in the paper) | [experiments/visualize_cross_attention.py](experiments/visualize_cross_attention.py) |
| Qualitative detection export (PLY) for MeshLab / CloudCompare | [experiments/visualize_detections.py](experiments/visualize_detections.py) |

---

## Key Findings

All detection numbers are on the ScanNetV2 validation split, using the official pretrained checkpoints, on one RTX 3090. SUN RGB-D was dropped as roughly 25-30% of val scenes failed in the MATLAB preprocessing scripts, which would have made the comparison unfair.

**1. Reproduction is faithful.** Within 1.2 AP25 of the paper; the residual gap is attributable to FPS randomness (acknowledged in the official repo).

| Model | AP25 (ours / paper) | AP50 (ours / paper) |
|---|---|---|
| 3DETR | 61.69 / 62.7 | 37.77 / 37.5 |
| 3DETR-m | 64.42 / 65.0 | 45.84 / 47.0 |

**2. The NMS-robustness claim holds.** Removing NMS costs 3DETR under 1 AP25 (60.82 vs 61.69), whereas VoteNet collapses from 58.27 to 9.76. Set-based bipartite matching really does suppress duplicates.

**3. Query budget matters more than the paper suggests.** At 32 queries 3DETR gets 13.16 AP25 (VoteNet with 32 proposals: 44.03); it needs at least 256 queries to match VoteNet, and it is slower at every setting (147 ms vs 121 ms at 256). The "adapt compute at inference" advantage comes with a real accuracy/latency trade-off.

| Queries / proposals | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| 3DETR AP25 | 13.16 | 34.25 | 53.06 | 61.69 | 63.98 |
| VoteNet AP25 | 44.03 | 52.75 | 57.32 | 58.27 | 57.67 |

**4. The encoder transfers to classification, at a cost.** ModelNet40 after 250 epochs (best at epoch 212): 89.8 OA / 88.3 mAcc, vs 92.1 / 89.1 reported. It beats PointNet on mAcc and is competitive with DGCNN, but one epoch takes 212 s vs 9 s for PointNet. The gap to the paper is likely due to training details the authors did not specify, plus our shorter schedule.

**5. Small objects are the weak spot.** Binning GT boxes into equal-frequency bins shows a sharp recall drop in the two smallest volume bins and a plateau beyond; binning by point count shows the same trend with a smaller gap. Our hypotheses: IoU sensitivity on small boxes, early point downsampling, FPS query coverage, and cross-attention dominated by nearby large structures. This is also why we expect poor transfer to sparse outdoor scenes.

**6. Queries do not attend locally.** Averaging cross-attention over the last three decoder layers, correctly localized objects do get high attention on their own points, but high values are also spread over walls, doors and unrelated regions. In the worst case (a detected toilet) attention is almost entirely outside the object. This lack of locality is a plausible inductive-bias gap, and is the issue later work such as V-DETR targets.

**7. Qualitative failure modes.** False positives on out-of-vocabulary objects (a basket detected as a garbage bin), spurious boxes in unscanned regions, and missed doors/windows/wall-mounted objects. Details and figures are in the [report](docs/report.pdf).

---

## What Changed vs. Upstream

**New files**
- [inference.py](inference.py): lightweight inference entry point (dataset, model and AP-config construction).
- [experiments/](experiments/): the five analysis scripts listed above; all reuse the same predictions or checkpoint and write CSVs / plots to `--output_dir`.
- [models/model_3detr_cls.py](models/model_3detr_cls.py): encoder-only classifier. No decoder or queries; a 2-layer MLP head on the CLS token (3DETR) or max-pooled features (3DETR-m).
- [datasets/modelnet40.py](datasets/modelnet40.py): normal-resampled ModelNet40, 8192 points, normalized; rotation / jitter / scaling augmentation at train time and a fixed per-index sampling at test time for reproducibility.
- [data_helper/](data_helper/): ScanNet download scripts (Python and PowerShell) and `rotate_scans.py`, which axis-aligns the raw `*_vh_clean_2.ply` meshes so they match the preprocessed boxes for visualization.
- [requirements.txt](requirements.txt): extra dependencies for visualization (`plyfile`, `matplotlib`, `trimesh`, `scipy`).
- [votenet/](votenet/): copy of the official [VoteNet](https://github.com/facebookresearch/votenet) used for the VoteNet comparisons.

**Modified files**
- [utils/ap_calculator.py](utils/ap_calculator.py): tracks per-GT-box volume and point count; adds `compute_recall_by_size_bin`, `compute_recall_by_num_points_bin`, `compute_recall_by_density_bin`.
- [datasets/scannet.py](datasets/scannet.py): returns `gt_num_points` per box; explicit `int32` dtype for `sem_seg_labels`.
- [utils/pc_util.py](utils/pc_util.py): PLY and oriented-box export helpers (`write_ply`, `write_ply_rgb`, `write_ply_color`, `write_oriented_bbox_cylinder`).
- [engine.py](engine.py), [main.py](main.py): `train_one_epoch_cls` / `evaluate_cls` (OA, mAcc), `do_train_cls` / `test_model_cls`, `--model_name 3detr_cls`, `--use_normals`, per-epoch timing, `weights_only=False` in `torch.load`.
- [datasets/__init__.py](datasets/__init__.py), [models/__init__.py](models/__init__.py): register `modelnet40` and `3detr_cls`.
- [models/transformer.py](models/transformer.py): dropout `inplace=True` -> `False` in the encoder layer.


---

## How to Run Our Additions

Setup, dataset preparation and pretrained weights follow the [original README](#original-3detr-readme) below. Extra dependencies: `pip install -r requirements.txt`. Dataset paths are set at the top of `datasets/scannet.py` and `datasets/modelnet40.py`.

All experiment scripts take the usual `main.py` flags (`--dataset_name`, `--test_ckpt`, model settings matching the checkpoint) plus the options below.

```bash
# Number of queries sweep (AP25 / AP50 / inference time)
python experiments/query_effect_analysis.py --dataset_name scannet \
    --test_ckpt scannet_ep1080.pth --query_counts 32 64 128 256 512 \
    --output_dir results/query_sweep

# NMS ablation (none / 2D / 3D class-agnostic / 3D per-class)
python experiments/nms_effect_analysis.py --dataset_name scannet \
    --test_ckpt scannet_ep1080.pth --output_dir results/nms_sweep

# Recall by object volume (--mode size | num_points | density)
python experiments/binned_recall_analysis.py --dataset_name scannet \
    --test_ckpt scannet_ep1080.pth --mode size --output_dir results/size_analysis

# Cross-attention maps for one scene (PLY output)
python experiments/visualize_cross_attention.py --dataset_name scannet \
    --test_ckpt scannet_ep1080.pth --scene_idx 197 --output_dir attn_results/

# Qualitative detections for one scene (PLY output)
python experiments/visualize_detections.py --dataset_name scannet \
    --test_ckpt scannet_ep1080.pth --scene_idx 16 --dump_dir last_qualitative/

# ModelNet40 classification with the 3DETR encoder
python main.py --dataset_name modelnet40 --dataset_root_dir <modelnet40_dir> \
    --model_name 3detr_cls --use_normals
```

Visualization needs the rotated ScanNet meshes: run `data_helper/rotate_scans.py` once first. PLY files open in MeshLab or CloudCompare. Batch wrappers over several scenes are in [scripts/predictions_visualization.sh](scripts/predictions_visualization.sh) and [scripts/cross_attention_visualization.sh](scripts/cross_attention_visualization.sh).

Classification hyperparameters: AdamW, lr 4e-4, weight decay 0.1, 20-epoch linear warmup from 4e-7, then decay to 4e-5 over the remaining 230 epochs. Detection runs use the repo defaults (3 encoder layers, 8 decoder layers, 256-d MLP).

---


# Original 3DETR README

The content below is the unmodified README from the upstream repository.

---

# 3DETR: An End-to-End Transformer Model for 3D Object Detection

PyTorch implementation and models for **3DETR**.

**3DETR** (**3D** **DE**tection **TR**ansformer) is a simpler alternative to complex hand-crafted 3D detection pipelines.
It does not rely on 3D backbones such as PointNet++ and uses few 3D-specific operators.
3DETR obtains comparable or better performance than 3D detection methods such as VoteNet.
The encoder can also be used for other 3D tasks such as shape classification.
More details in the paper ["An End-to-End Transformer Model for 3D Object Detection"](http://arxiv.org/abs/2109.08141).

[[`website`](https://facebookresearch.github.io/3detr)] [[`arXiv`](http://arxiv.org/abs/2109.08141)] [[`bibtex`](#Citation)]

**Code description.** Our code is based on prior work such as DETR and VoteNet and we aim for simplicity in our implementation. We hope it can ease research in 3D detection.

![3DETR Approach](.github/approach.jpg)
![Decoder Detections](.github/decoder_detections.jpg)

# Pretrained Models

We provide the pretrained model weights and the corresponding metrics on the val set (per class APs, Recalls).
We provide a Python script [`utils/download_weights.py`](utils/download_weights.py) to easily download the weights/metrics files.

<table>
<tr>
<th>Arch</th>
<th>Dataset</th>
<th>Epochs</th>
<th>AP25</th>
<th>AP50</th>
<th>Model weights</th>
<th>Eval metrics</th>
</tr>
<tr>
<td>3DETR-m</td>
<td>SUN RGB-D</td>
<td>1080</td>
<td>59.1</td>
<td>30.3</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_masked_ep1080.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_masked_ep1080_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR</td>
<td>SUN RGB-D</td>
<td>1080</td>
<td>58.0</td>
<td>30.3</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_ep1080.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_ep1080_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR-m</td>
<td>ScanNet</td>
<td>1080</td>
<td>65.0</td>
<td>47.0</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_masked_ep1080.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_masked_ep1080_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR</td>
<td>ScanNet</td>
<td>1080</td>
<td>62.1</td>
<td>37.9</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_ep1080.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_ep1080_metrics.pkl">metrics</a></td>
</tr>
</table>

## Model Zoo

For convenience, we provide model weights for 3DETR trained for different number of epochs.

<table>
<tr>
<th>Arch</th>
<th>Dataset</th>
<th>Epochs</th>
<th>AP25</th>
<th>AP50</th>
<th>Model weights</th>
<th>Eval metrics</th>
</tr>
<tr>
<td>3DETR-m</td>
<td>SUN RGB-D</td>
<td>90</td>
<td>51.0</td>
<td>22.0</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_masked_ep90.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_masked_ep90_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR-m</td>
<td>SUN RGB-D</td>
<td>180</td>
<td>55.6</td>
<td>27.5</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_masked_ep180.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_masked_ep180_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR-m</td>
<td>SUN RGB-D</td>
<td>360</td>
<td>58.2</td>
<td>30.6</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_masked_ep360.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_masked_ep360_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR-m</td>
<td>SUN RGB-D</td>
<td>720</td>
<td>58.1</td>
<td>30.4</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_masked_ep720.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_masked_ep720_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR</td>
<td>SUN RGB-D</td>
<td>90</td>
<td>43.7</td>
<td>16.2</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_ep90.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_ep90_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR</td>
<td>SUN RGB-D</td>
<td>180</td>
<td>52.1</td>
<td>25.8</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_ep180.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_ep180_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR</td>
<td>SUN RGB-D</td>
<td>360</td>
<td>56.3</td>
<td>29.6</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_ep360.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_ep360_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR</td>
<td>SUN RGB-D</td>
<td>720</td>
<td>56.0</td>
<td>27.8</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_ep720.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/sunrgbd_ep720_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR-m</td>
<td>ScanNet</td>
<td>90</td>
<td>47.1</td>
<td>19.5</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_masked_ep90.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_masked_ep90_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR-m</td>
<td>ScanNet</td>
<td>180</td>
<td>58.7</td>
<td>33.6</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_masked_ep180.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_masked_ep180_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR-m</td>
<td>ScanNet</td>
<td>360</td>
<td>62.4</td>
<td>37.7</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_masked_ep360.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_masked_ep360_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR-m</td>
<td>ScanNet</td>
<td>720</td>
<td>63.7</td>
<td>44.5</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_masked_ep720.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_masked_ep720_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR</td>
<td>ScanNet</td>
<td>90</td>
<td>42.8</td>
<td>15.3</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_ep90.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_ep90_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR</td>
<td>ScanNet</td>
<td>180</td>
<td>54.5</td>
<td>28.8</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_ep180.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_ep180_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR</td>
<td>ScanNet</td>
<td>360</td>
<td>59.0</td>
<td>35.4</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_ep360.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_ep360_metrics.pkl">metrics</a></td>
</tr>
<tr>
<td>3DETR</td>
<td>ScanNet</td>
<td>720</td>
<td>61.1</td>
<td>40.2</td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_ep720.pth">weights</a></td>
<td><a href="https://dl.fbaipublicfiles.com/3detr/checkpoints/scannet_ep720_metrics.pkl">metrics</a></td>
</tr>
</table>


# Running 3DETR

## Installation
Our code is tested with PyTorch 1.9.0, CUDA 10.2 and Python 3.6. It may work with other versions.

You will need to install `pointnet2` layers by running

```
cd third_party/pointnet2 && python setup.py install
```

You will also need Python dependencies (either `conda install` or `pip install`)

```
matplotlib
opencv-python
plyfile
'trimesh>=2.35.39,<2.35.40'
'networkx>=2.2,<2.3'
scipy
```

Some users have experienced issues using CUDA 11 or higher. Please try using CUDA 10.2 if you run into CUDA issues.

**Optionally**, you can install a Cythonized implementation of gIOU for faster training.
```
conda install cython
cd utils && python cython_compile.py build_ext --inplace
```


# Benchmarking

## Dataset preparation

We follow the VoteNet codebase for preprocessing our data.
The instructions for preprocessing SUN RGB-D are [here](https://github.com/facebookresearch/votenet/tree/main/sunrgbd) and ScanNet are [here](https://github.com/facebookresearch/votenet/tree/main/scannet).

You can edit the dataset paths in [`datasets/sunrgbd.py`](datasets/sunrgbd.py#L36) and [`datasets/scannet.py`](datasets/scannet.py#L23-L24) or choose to specify at runtime.

## Testing

Once you have the datasets prepared, you can test pretrained models as

```
python main.py --dataset_name <dataset_name> --nqueries <number of queries> --test_ckpt <path_to_checkpoint> --test_only [--enc_type masked]
```

We use 128 queries for the SUN RGB-D dataset and 256 queries for the ScanNet dataset.
You will need to add the flag `--enc_type masked` when testing the 3DETR-m checkpoints.
Please note that the testing process is stochastic (due to randomness in point cloud sampling and sampling the queries) and so results can vary within 1% AP25 across runs.
This stochastic nature of the inference process is also common for methods such as VoteNet.

If you have not edited the dataset paths for the files in the `datasets` folder, you can pass the path to the datasets using the `--dataset_root_dir` flag.

## Training

The model can be simply trained by running `main.py`.
```
python main.py --dataset_name <dataset_name> --checkpoint_dir <path to store outputs>
```

To reproduce the results in the paper, we provide the arguments in the [`scripts`](scripts/) folder.
A variance of 1% AP25 across different training runs can be expected.

You can quickly verify your installation by training a 3DETR model for 90 epochs on ScanNet following the file `scripts/scannet_quick.sh` and compare it to the pretrained checkpoint from the Model Zoo.


## License
The majority of 3DETR is licensed under the Apache 2.0 license as found in the [LICENSE](LICENSE) file, however portions of the project are available under separate license terms: licensing information for pointnet2 is available at https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/UNLICENSE

## Contributing
We welcome your pull requests! Please see [CONTRIBUTING](CONTRIBUTING.md) and [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md) for more info.

## Citation
If you find this repository useful, please consider starring :star: us and citing

```
@inproceedings{misra2021-3detr,
    title={{An End-to-End Transformer Model for 3D Object Detection}},
    author={Misra, Ishan and Girdhar, Rohit and Joulin, Armand},
    booktitle={{ICCV}},
    year={2021},
}
```
