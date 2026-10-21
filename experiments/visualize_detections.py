import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from models import build_model
from datasets import build_dataset
from utils.ap_calculator import parse_predictions, get_ap_config_dict
import utils.pc_util as pc_util
import main

# Color defining
_CLASS_COLORS = np.array([
    [230,  25,  75], [ 60, 180,  75], [255, 225,  25], [  0, 130, 200],
    [245, 130,  48], [145,  30, 180], [ 70, 240, 240], [240,  50, 230],
    [210, 245,  60], [250, 190, 212], [  0, 128, 128], [220, 190, 255],
    [170, 110,  40], [255, 250, 200], [128,   0,   0], [170, 255, 195],
    [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128],
], dtype=np.uint8)


# Visualization logic
def dump_results(end_points, dump_dir, conf_thresh):
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    objectness_prob = end_points['objectness_prob'].detach().cpu().numpy()
    pred_center = end_points['center_unnormalized'].detach().cpu().numpy() 
    pred_size = end_points['size_unnormalized'].detach().cpu().numpy()
    pred_angle = end_points['angle_continuous'].detach().cpu().numpy()
    query_xyz = end_points['query_xyz'].detach().cpu().numpy()
    sem_cls_prob = end_points['sem_cls_prob'].detach().cpu().numpy()
    pred_cls = sem_cls_prob.argmax(axis=-1)

    for i in range(batch_size):
        # ORIGINAL POINT CLOUD
        vh_2_clean_file = os.path.join(f"/workspace/3DETR/scans/rotated_vh_clean_2_files/{end_points['scene_name']}_vh_clean_2.ply")
        if os.path.isfile(vh_2_clean_file):
            import shutil
            shutil.copy(vh_2_clean_file, os.path.join(dump_dir, 'scene_original_vh_clean_2.ply'))

        pc = point_clouds[i, :, 0:3]
        obj_prob = objectness_prob[i]

        # pc_util.write_ply_rgb(pc, end_points['pcl_color'][i], os.path.join(dump_dir, f'pc.ply'))
        # pc_util.write_ply(query_xyz[i], os.path.join(dump_dir, f'query_pc.ply'))
        # if np.sum(obj_prob > conf_thresh) > 0:
            # pc_util.write_ply(pred_center[i, obj_prob > conf_thresh, :], os.path.join(dump_dir, f'{end_points['enc_type']}_confident_proposal_pc.ply'))

        # obbs = np.hstack([pred_center[i], pred_size[i], pred_angle[i].reshape(-1, 1)])  # (K, 7)

        # pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, f'pred_bbox.ply'))
        # if np.sum(obj_prob > conf_thresh) > 0:
        #     pc_util.write_oriented_bbox(obbs[obj_prob > conf_thresh], os.path.join(dump_dir, f'{end_points['enc_type']}_pred_confident_bbox.ply'))
        # if 'nms_mask' in end_points and np.sum(end_points['nms_mask'][i]) > 0:
        #     pc_util.write_oriented_bbox(obbs[end_points['nms_mask'][i]], os.path.join(dump_dir, f'{end_points['enc_type']}_pred_nms_bbox.ply'))

        obbs = np.hstack([pred_center[i], pred_size[i], pred_angle[i].reshape(-1, 1)])  # (K, 7)
        box_colors = _CLASS_COLORS[pred_cls[i] % len(_CLASS_COLORS)]                    # (K, 3)

        if np.sum(obj_prob > conf_thresh) > 0:
            mask = obj_prob > conf_thresh
            # pc_util.write_oriented_bbox_cylinder(obbs[mask], os.path.join(dump_dir, f'{end_points['enc_type']}_pred_confident_bbox.ply'), colors=box_colors[mask])
        if 'nms_mask' in end_points and np.sum(end_points['nms_mask'][i]) > 0:
            mask = end_points['nms_mask'][i]
            pc_util.write_oriented_bbox_cylinder(obbs[mask], os.path.join(dump_dir, f'{end_points['enc_type']}_pred_nms_bbox.ply'), colors=box_colors[mask])

    # LABELS AND OTHERS
    gt_center = end_points['gt_box_centers']
    gt_size = end_points['gt_box_sizes']
    gt_angle = end_points['gt_box_angles']
    gt_mask = end_points['gt_box_present']
    gt_cls = end_points['gt_box_sem_cls_label']

    for i in range(batch_size):
        obbs = []
        colors = []
        for j in range(gt_center.shape[1]):
            if gt_mask[i, j] == 0:
                continue
            obb = np.concatenate([gt_center[i, j], gt_size[i, j], [gt_angle[i, j]]])
            obbs.append(obb)
            colors.append(_CLASS_COLORS[int(gt_cls[i, j]) % len(_CLASS_COLORS)])
        if len(obbs) > 0:
            obbs = np.vstack(obbs)
            pc_util.write_oriented_bbox_cylinder(obbs, os.path.join(dump_dir, 'gt_bbox.ply'), colors=colors)


# Inference 
@torch.no_grad()
def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build test dataset and load scene
    datasets, dataset_config = build_dataset(args)
    dataset = datasets['test']
    sample = dataset[args.scene_idx]
    print('Loaded scene %d | %d points' % (args.scene_idx, sample['point_clouds'].shape[0]))

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

    # Run forward
    with torch.no_grad():
        outputs = model(inputs)['outputs']

    ap_config_dict = get_ap_config_dict(dataset_config=dataset_config, conf_thresh=args.conf_thresh, use_3d_nms=args.use_3d_nms)
    pred_map_cls = parse_predictions(
        predicted_boxes=outputs["box_corners"],
        sem_cls_probs=outputs["sem_cls_prob"],
        objectness_probs=outputs["objectness_prob"],
        point_cloud=inputs["point_clouds"],
        config_dict=ap_config_dict,
    )
    print('Finished detection with %d objects detected.' % len(pred_map_cls[0]))

    # Process NMS
    corners_all = outputs['box_corners'][0].cpu().numpy()
    nms_mask = np.zeros(corners_all.shape[0], dtype=bool)
    for _, det_corners, _ in pred_map_cls[0]:
        idx = np.abs(corners_all - det_corners[None]).sum(axis=(1, 2)).argmin()
        nms_mask[idx] = True

    scene_name = dataset.scan_names[args.scene_idx]

    return scene_name, inputs, outputs, sample, nms_mask

def make_args_parser():
    parser = main.make_args_parser()
    parser.add_argument('--scene_idx', default=0, type=int)
    parser.add_argument('--conf_thresh', default=0.5, type=float)
    parser.add_argument('--dump_dir', default='demo_results', type=str)
    parser.add_argument('--use_3d_nms', default=True, type=bool)
    return parser


if __name__ == '__main__':
    parser = make_args_parser()
    args = parser.parse_args()
    
    scene_name, inputs, output, gt_sample, nms_mask = run_inference(args)

    # Collect data for visualization
    end_points = {
        'scene_name': scene_name,
        'enc_type': args.enc_type,
        'point_clouds': inputs['point_clouds'],
        'objectness_prob': output['objectness_prob'][0].unsqueeze(0),
        'center_unnormalized': output['center_unnormalized'][0].unsqueeze(0),
        'size_unnormalized': output['size_unnormalized'][0].unsqueeze(0),
        'angle_continuous': output['angle_continuous'][0].unsqueeze(0),
        'query_xyz': output.get('query_xyz', output['center_unnormalized'])[0].unsqueeze(0),
        'pcl_color': (gt_sample['pcl_color'] * 256 + 128).clip(0, 255)[None] if args.use_color else gt_sample['pcl_color'][None],
        'gt_box_centers': gt_sample['gt_box_centers'][None],
        'gt_box_sizes': gt_sample['gt_box_sizes'][None],
        'gt_box_angles': gt_sample['gt_box_angles'][None],
        'gt_box_present': gt_sample['gt_box_present'][None],
        'gt_box_sem_cls_label': gt_sample['gt_box_sem_cls_label'][None],
        'nms_mask': nms_mask[None],
        'sem_cls_prob': output['sem_cls_prob'][0].unsqueeze(0),
    }

    dump_dir = os.path.join(args.dump_dir, 'scene%04d' % args.scene_idx)
    dump_results(end_points, dump_dir, args.conf_thresh)
    print('Dumped results to %s' % dump_dir)