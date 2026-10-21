import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from utils.ap_calculator import APCalculator, get_ap_config_dict
from utils.misc import my_worker_init_fn

import main


def make_ap_configs(dataset_config):
    return [
        ("No NMS", get_ap_config_dict(dataset_config=dataset_config, no_nms=True, use_3d_nms=False, cls_nms=False)),
        ("2D NMS", get_ap_config_dict(dataset_config=dataset_config, no_nms=False, use_3d_nms=False, cls_nms=False)),
        ("3D NMS class-agnostic", get_ap_config_dict(dataset_config=dataset_config, no_nms=False, use_3d_nms=True, cls_nms=False)),
        ("3D NMS per-class", get_ap_config_dict(dataset_config=dataset_config, no_nms=False, use_3d_nms=True, cls_nms=True))
    ]

def run_sweep(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Collect configs and build calculator for each
    configs = make_ap_configs(dataset_config)
    calculators = [
        APCalculator(
            dataset_config=dataset_config,
            ap_iou_thresh=[0.25, 0.5],
            class2type_map=dataset_config.class2type,
            exact_eval=True,
            ap_config_dict=cfg,
        )
        for name, cfg in configs
    ]

    # Run forward pass
    with torch.no_grad():
        for batch_idx, batch_data_label in enumerate(dataloader):
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(device)

            inputs = {
                "point_clouds": batch_data_label["point_clouds"],
                "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
                "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
            }
            outputs = model(inputs)["outputs"]

            for calculator in calculators:
                calculator.step_meter(outputs, batch_data_label)

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch [{batch_idx + 1}/{len(dataloader)}]")

    results = []
    for (label, _), calc in zip(configs, calculators):
        metrics = calc.compute_metrics()
        ap25 = metrics.get(0.25, {}).get("mAP", float("nan")) * 100
        ap50 = metrics.get(0.50, {}).get("mAP", float("nan")) * 100
        results.append((label, ap25, ap50, metrics))

    
    # First csv: summary of mAP
    save_csv(results, args.output_dir)

    class_names = list(dataset_config.class2type.values())
    labels = [label for label, *_ in results]

    # # Second csv: per-class AP@0.25
    # ap25_path = os.path.join(out_dir, "nms_sweep_ap25.csv")
    # with open(ap25_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["class"] + labels)
    #     for cls_name in class_names:
    #         writer.writerow([cls_name] + [f"{metrics.get(0.25, {}).get(cls_name, float('nan')) * 100:.1f}" for _, _, _, metrics in results])
    
    # # Third csv: per-class AP@0.50
    # ap50_path = os.path.join(out_dir, "nms_sweep_ap50.csv")
    # with open(ap50_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["class"] + labels)
    #     for cls_name in class_names:
    #         writer.writerow([cls_name] + [f"{metrics.get(0.50, {}).get(cls_name, float('nan')) * 100:.1f}" for _, _, _, metrics in results])

    # print(f"Sweep completed. Results saved to:\n{summary_path}\n{ap25_path}\n{ap50_path}")

def save_csv(results, output_dir):
    with open(os.path.join(output_dir, "nms_sweep_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nms_config", "mAP@0.25", "mAP@0.50"])
        for label, ap25, ap50, _ in results:
            writer.writerow([label, f"{ap25:.2f}", f"{ap50:.2f}"])
    # print("Saved csv.")
    print(f"\nResults saved to: {output_dir}/")

def make_args_parser():
    parser = main.make_args_parser()
    parser.add_argument("--output_dir", default="results/nms_sweep", type=str)
    return parser

if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    args.test_only = True

    run_sweep(args)
