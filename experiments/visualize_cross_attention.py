import os
import numpy as np
import torch
import matplotlib.cm as cm
from scipy.spatial import cKDTree

from models import build_model
from datasets import build_dataset
import main
import utils.pc_util as pc_util
from utils.ap_calculator import parse_predictions, get_ap_config_dict


def forward_with_attention(model, inputs):
    # Same as forward for 3DETR, but also returns decoder attention
    model.eval()
    with torch.no_grad():
        point_clouds = inputs["point_clouds"]

        enc_xyz, enc_features, _ = model.run_encoder(point_clouds)
        enc_features = model.encoder_to_decoder_projection(enc_features.permute(1, 2, 0)).permute(2, 0, 1)

        point_cloud_dims = [inputs["point_cloud_dims_min"], inputs["point_cloud_dims_max"]]
        query_xyz, query_embed = model.get_query_embeddings(enc_xyz, point_cloud_dims)
        enc_pos = model.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)

        box_features, attns = model.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos,
            return_attn_weights=True,
        )

        box_predictions = model.get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )

    return box_predictions, enc_xyz, attns


def visualize_attention(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build test dataset and load scene
    datasets, dataset_config = build_dataset(args)
    dataset = datasets["test"]
    sample = dataset[args.scene_idx]
    print('Loaded scene %d | %d points' % (args.scene_idx, sample['point_clouds'].shape[0]))

    # Make output dir
    scene_name = dataset.scan_names[args.scene_idx]
    os.makedirs(os.path.join(args.output_dir, scene_name), exist_ok=True)

    # Build model
    model, _ = build_model(args, dataset_config)
    model = model.to(device)
    checkpoint = torch.load(args.test_ckpt, map_location=device)
    model.load_state_dict(checkpoint.get('model', checkpoint))
    model.eval()
    print('Loaded checkpoint %s' % args.test_ckpt)

    inputs = {
        'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).float().to(device),
        'point_cloud_dims_min': torch.from_numpy(sample['point_cloud_dims_min']).unsqueeze(0).float().to(device),
        'point_cloud_dims_max': torch.from_numpy(sample['point_cloud_dims_max']).unsqueeze(0).float().to(device),
    }

    # Run forward and get attention
    box_predictions, enc_xyz, attns = forward_with_attention(model, inputs)

    # Extract output
    outputs = box_predictions["outputs"]
    center = outputs["center_unnormalized"][0].cpu().numpy()
    size = outputs["size_unnormalized"][0].cpu().numpy()
    angle = outputs["angle_continuous"][0].cpu().numpy()

    # Apply empty-box removal + NMS (same pipeline as eval)
    config_dict = get_ap_config_dict(
        dataset_config=dataset_config,
        remove_empty_box=True,
        use_3d_nms=True,
        nms_iou=0.25,
        cls_nms=True,
        per_class_proposal=False,
        conf_thresh=args.obj_probs_threshold,
    )
    batch_pred = parse_predictions(
        outputs["box_corners"],
        outputs["sem_cls_prob"],
        outputs["objectness_prob"],
        inputs["point_clouds"],
        config_dict,
    )
    preds = sorted(batch_pred[0], key=lambda x: x[2], reverse=True)
    print(f"Found {len(preds)} detections after NMS.")
    if len(preds) == 0:
        return

    # Map each surviving prediction back to its original query index (needed for attention lookup)
    box_corners_np = outputs["box_corners"][0].cpu().numpy()  # (K, 8, 3)
    keep = [
        int(np.argmin(np.abs(box_corners_np - pred_corners[None]).sum(axis=(1, 2))))
        for _, pred_corners, _ in preds
    ]

    # Verify: each kept query's corners must exactly match the NMS output
    for q_idx, (_, pred_corners, _) in zip(keep, preds):
        assert np.allclose(box_corners_np[q_idx], pred_corners), \
            f"Corner mismatch at query {q_idx}"

    # Last decoder layer attention extraction with points
    # dec_attn = attns[-1, 0].cpu().numpy()

    dec_attn = attns[-3:].mean(dim=0)[0].cpu().numpy()
    enc_pts = enc_xyz[0].cpu().numpy() # (N_enc, 3)
    full_pc = sample["point_clouds"][:, :3] # (N, 3)
    print(f"Encoder points: {enc_pts.shape}, attention shape: {dec_attn.shape}")

    print(attns.shape)
    # Find NN for interpolation
    tree = cKDTree(enc_pts)
    _, nn_idx = tree.query(full_pc, k=1)

    for q_idx in keep:
        attn = dec_attn[q_idx]
        # Normalize attention so visualization is not dominated by extreme values
        p_lo, p_hi = np.percentile(attn, [2, 98])
        attn_norm = np.clip((attn - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)

        # Propagate attention to full pc
        full_attn = attn_norm[nn_idx]
        rgba = cm.plasma(full_attn)
        # rgba = cm.viridis(full_attn)
        rgb = (rgba[:, :3] * 255).astype(np.uint8)

        # Save attention colored pc
        pc_util.write_ply_rgb(full_pc, rgb, os.path.join(args.output_dir, scene_name, f"query_{q_idx}_attn.ply"))

        # Save predicted bb
        obb = np.hstack([center[q_idx], size[q_idx], [angle[q_idx]]])[None]
        pc_util.write_oriented_bbox_cylinder(obb, os.path.join(args.output_dir, scene_name, f"query_{q_idx}_bbox.ply"))

    # ORIGINAL POINT CLOUD
    vh_2_clean_file = os.path.join(f"/workspace/3DETR/scans/rotated_vh_clean_2_files/{scene_name}_vh_clean_2.ply")
    print(vh_2_clean_file)
    if os.path.isfile(vh_2_clean_file):
        print("no")
        import shutil
        shutil.copy(vh_2_clean_file, os.path.join(args.output_dir, scene_name, 'scene_original_vh_clean_2.ply'))

    print("Processed and saved attention files.")

def make_args_parser():
    parser = main.make_args_parser()
    parser.add_argument("--scene_idx", default=0, type=int)
    parser.add_argument("--obj_probs_threshold", default=0.5, type=float)
    parser.add_argument("--output_dir", default="attn_results", type=str)
    return parser


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    visualize_attention(args)
