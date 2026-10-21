import os
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import build_dataset
from models import build_model
from utils.ap_calculator import APCalculator
from utils.misc import my_worker_init_fn
import main

# Global bin definitions
SIZE_BINS = [
    (0.000, 0.041),
    (0.041, 0.097),
    (0.097, 0.152),
    (0.152, 0.215),
    (0.215, 0.285),
    (0.285, 0.366),
    (0.366, 0.487),
    (0.487, 0.769),
    (0.769, 1.411),
    (1.411, float("inf")),
]
SIZE_BIN_NAMES = [
    f"{lo:g}-{'inf' if hi == float('inf') else f'{hi:g}'}"
    for lo, hi in SIZE_BINS
]

POINT_BINS = [
    (0, 172),
    (172, 309),
    (309, 439),
    (439, 595),
    (595, 816),
    (816, 1123),
    (1123, 1613),
    (1613, 2512),
    (2512, 4523),
    (4523, float("inf")),
]
PT_BIN_NAMES = [
    f"{lo:g}-{'inf' if hi == float('inf') else f'{hi:g}'}"
    for lo, hi in POINT_BINS
]

DENSITY_BINS = [
    (0, float("inf")),
    (0, 354),
    (354, 784),
    (784, 1343),
    (1343, 2044),
    (2044, 2970),
    (2970, 4420),
    (4420, 7157),
    (7157, 12699),
    (12699, 37147),
    (37147, float("inf")),
]
DENSITY_BIN_NAMES = [
    f"{lo:g}-{'inf' if hi == float('inf') else f'{hi:g}'}"
    for lo, hi in DENSITY_BINS
]

# Inference and file savings
@torch.no_grad()
def run_inference(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build test dataset
    datasets, dataset_config = build_dataset(args)
    test_dataset = datasets["test"]
    sampler = torch.utils.data.SequentialSampler(test_dataset)
    dataloader = DataLoader(
        test_dataset,
        sampler=sampler,
        batch_size=args.batchsize_per_gpu,
        num_workers=args.dataset_num_workers,
        worker_init_fn=my_worker_init_fn,
    )

    # Build model and load checkpoint
    model, _ = build_model(args, dataset_config)
    model = model.to(device)
    checkpoint = torch.load(args.test_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print('Loaded checkpoint %s' % args.test_ckpt)

    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )
    
    # Run forward
    model.eval()
    for batch_idx, batch_data_label in enumerate(dataloader):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        outputs = model(inputs)["outputs"]

        ap_calculator.step_meter(outputs, batch_data_label)
        
        if (batch_idx+1) % 10 == 0:
            print(f"Batch [{batch_idx + 1}/{len(dataloader)}]")

    print(f"\nFinished inference.")
    return ap_calculator, dataset_config


def save_csv(recall_by_iou, bin_names, out_dir, mode):
    path = os.path.join(out_dir, f"recall_by_{mode}.csv")
    iou_thresholds = sorted(recall_by_iou.keys())

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["bin"]
        for iou in iou_thresholds:
            header += [f"recall@{iou:.2f} (%)", f"total_gt@{iou:.2f}"]
        writer.writerow(header)

        for bin_label, bin_name in zip(recall_by_iou[iou_thresholds[0]].keys(), bin_names):
            row = [bin_name]
            for iou in iou_thresholds:
                recall = recall_by_iou[iou][bin_label]
                row += [f"{recall['overall_recall'] * 100:.2f}", str(recall["total_gt"])]
            writer.writerow(row)


def plot_results(recall_by_iou, bin_names, out_dir, mode):
    path = os.path.join(out_dir, f"recall_by_{mode}.png")

    iou_thresholds = sorted(recall_by_iou.keys())
    bin_labels = list(recall_by_iou[iou_thresholds[0]].keys())
    display_names = bin_names if len(bin_names) == len(bin_labels) else bin_labels
    n_bins, n_iou = len(bin_labels), len(iou_thresholds)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_iou))
    width = 0.8 / n_iou
    x = np.arange(n_bins)
    fig, ax = plt.subplots(figsize=(max(6, n_bins * 1.5), 4))

    for i, (iou, color) in enumerate(zip(iou_thresholds, colors)):
        recalls = [recall_by_iou[iou].get(bl, {}).get("overall_recall", 0) * 100 for bl in bin_labels]
        bars = ax.bar(x + i * width, recalls, width, label=f"IoU={iou:.2f}", color=color)
        for bar, v in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (n_iou - 1) / 2)
    ax.set_xticklabels(display_names, rotation=15, ha="right")
    OVERALL_RECALL_25 = 76.5
    OVERALL_RECALL_50 = 51.6
    ax.axhline(OVERALL_RECALL_25, color="red",   linestyle="--", linewidth=1.2, label=f"Overall IoU=0.25 ({OVERALL_RECALL_25}%)")
    ax.axhline(OVERALL_RECALL_50, color="orange", linestyle="--", linewidth=1.2, label=f"Overall IoU=0.50 ({OVERALL_RECALL_50}%)")

    title_map = {
        "size": "3DETR Recall by Object Volume",
        "num_points": "3DETR Recall by Points per Object",
        "density": "3DETR Recall by Object Density",
    }
    
    ax.set_ylabel("Recall (%)")
    ax.set_title(title_map[mode])
    ax.set_ylim(0, 110)
    ax.legend()
    fig.tight_layout()
    
    fig.savefig(path, dpi=150)
    plt.close(fig)

def make_args_parser():
    parser = main.make_args_parser()
    parser.add_argument("--output_dir", default="results/size_analysis", type=str)
    parser.add_argument("--mode", default="size", choices=["size", "num_points", "density"],
                        help="Bin by volume (size) or points per box (num_points)")
    return parser

if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    args.test_only = True

    ap_calculator, dataset_config = run_inference(args)

    overall = ap_calculator.compute_metrics()
    print("\nOverall Metrics")
    print(ap_calculator.metrics_to_str(overall, per_class=False))

    if args.mode == "size":
        bins, bin_names = SIZE_BINS, SIZE_BIN_NAMES
        compute_fn = ap_calculator.compute_recall_by_size_bin
    elif args.mode == "num_points":
        bins, bin_names = POINT_BINS, PT_BIN_NAMES
        compute_fn = ap_calculator.compute_recall_by_num_points_bin
    else:
        bins, bin_names = DENSITY_BINS, DENSITY_BIN_NAMES
        compute_fn = ap_calculator.compute_recall_by_density_bin

    print(f"\nComputing {args.mode}-binned recall.")
    recall_by_iou = {}
    for iou_thresh in [0.25, 0.5]:
        recall_by_iou[iou_thresh] = compute_fn(bins, iou_thresh=iou_thresh)

    save_csv(recall_by_iou, bin_names, args.output_dir, args.mode)
    plot_results(recall_by_iou, bin_names, args.output_dir, args.mode)

    print(f"\nResults saved to: {args.output_dir}/")

