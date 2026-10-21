import csv
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from utils.ap_calculator import APCalculator, get_ap_config_dict
from utils.misc import my_worker_init_fn
import main

@torch.no_grad()
def run_one(model, dataloader, dataset_config, device, num_queries):
    model.num_queries = num_queries

    ap_config_dict = get_ap_config_dict(dataset_config=dataset_config)
    calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
        ap_config_dict=ap_config_dict,
    )

    # Warmup steps
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:
            break

        inputs = {
            "point_clouds": batch["point_clouds"].to(device),
            "point_cloud_dims_min": batch["point_cloud_dims_min"].to(device),
            "point_cloud_dims_max": batch["point_cloud_dims_max"].to(device),
        }
        model(inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

    # Evaluate model for current query config
    timings = []
    for batch_idx, batch_data_label in enumerate(dataloader):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)

        calculator.step_meter(outputs, batch_data_label)

    metrics = calculator.compute_metrics()
    ap25 = metrics.get(0.25, {}).get("mAP", float("nan")) * 100
    ap50 = metrics.get(0.50, {}).get("mAP", float("nan")) * 100
    mean_ms = np.mean(timings) * 1000
    std_ms  = np.std(timings)  * 1000
    return ap25, ap50, mean_ms, std_ms


def run(args):
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

    print(f"Running sweep of query counts {args.query_counts}\n")
    results = []
    for num_queries in args.query_counts:
        print(f"Running inference for query={num_queries}.")
        ap25, ap50, mean_ms, std_ms = run_one(
            model, dataloader, 
            dataset_config, device,
            num_queries,
        )
        results.append((num_queries, ap25, ap50, mean_ms, std_ms))

    print("Finished sweeping.")
    save_csv(results, args.output_dir)

def save_csv(results, output_dir):
    with open(os.path.join(output_dir, "results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_queries", "mAP@0.25", "mAP@0.50", "mean_time_ms", "std_time_ms"])
        for num_queries, ap25, ap50, mean_ms, std_ms in results:
            writer.writerow([num_queries, f"{ap25:.2f}", f"{ap50:.2f}", f"{mean_ms:.2f}", f"{std_ms:.2f}"])
    # print("Saved csv.")
    print(f"\nResults saved to: {output_dir}/")

def make_args_parser():
    parser = main.make_args_parser()
    parser.add_argument("--output_dir", default="results/query_sweep", type=str)
    parser.add_argument("--query_counts", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    return parser

if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    args.test_only = True
    run(args)
