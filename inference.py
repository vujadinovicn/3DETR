import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from utils.misc import my_worker_init_fn
import utils.ap_calculator as ap_calculator
import main

def make_args_parser():
   return main.make_args_parser()

def get_ap_config_dict(args, dataset_config):
    return ap_calculator.get_ap_config_dict(
        dataset_config=dataset_config, remove_empty_box=not args.no_remove_empty_box,
        use_3d_nms=True, nms_iou=args.nms_iou, use_old_type_nms=args.use_old_type_nms,
        cls_nms=args.per_class_proposal, per_class_proposal=args.per_class_proposal,
        conf_thresh=args.conf_thresh, no_nms=False
    )

@torch.no_grad()
def run_inference(args):
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

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
    model, _ = build_model()
    model = model.to(device)
    checkpoint = torch.load(args.test_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print('Loaded checkpoint %s' % args.checkpoint)

    ap_config_dict = ap_calculator.get_ap_config_dict(args, dataset_config)

    all_predictions = []
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

            batch_pred_map_cls = ap_calculator.parse_predictions(
                predicted_boxes=outputs["box_corners"],
                sem_cls_probs=outputs["sem_cls_prob"],
                objectness_probs=outputs["objectness_prob"],
                point_cloud=batch_data_label["point_clouds"],
                config_dict=ap_config_dict,
            )
            all_predictions.extend(batch_pred_map_cls)

            print(f"Batch [{batch_idx + 1}/{len(dataloader)}]")

    print(f"\nFinished inference with {len(all_predictions)} samples processed.")
    return all_predictions


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    args.test_only = True

    predictions = run_inference(args)
