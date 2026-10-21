import os
import numpy as np
import utils.pc_util as pc_util
from torch.utils.data import Dataset

DATASET_ROOT_DIR = "/workspace/modelnet40_normal_resampled"


class ModelNet40DatasetConfig:
    def __init__(self):
        self.num_classes = 40
        self.class_names = [
            "airplane", "bathtub", "bed", "bench", "bookshelf",
            "bottle", "bowl", "car", "chair", "cone",
            "cup", "curtain", "desk", "door", "dresser",
            "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
            "laptop", "mantel", "monitor", "night_stand", "person",
            "piano", "plant", "radio", "range_hood", "sink",
            "sofa", "stairs", "stool", "table", "tent",
            "toilet", "tv_stand", "vase", "wardrobe", "xbox",
        ]
        self.class2idx = {c: i for i, c in enumerate(self.class_names)}
        self.idx2class = {i: c for i, c in enumerate(self.class_names)}


class ModelNet40ClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        root_dir=None,
        num_points=8192,
        use_normals=True,
        augment=False,
    ):
        assert num_points <= 10000, "Pre-sampled shapes have a maximum of 10000 points"
        assert split_set in ["train", "test"]
        self.dataset_config = dataset_config

        if root_dir is None:
                root_dir = DATASET_ROOT_DIR
        self.root_dir = root_dir

        self.num_points = num_points
        self.use_normals = use_normals
        self.augment = augment

        split_file = os.path.join(root_dir, f"modelnet40_{split_set}.txt")
        with open(split_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.samples = []
        for line in lines:
            sample = line.split()[0]

            class_name = sample.rsplit("_", 1)[0]
            label = dataset_config.class2idx[class_name]
            
            filepath = os.path.join(root_dir, class_name, sample + ".txt")
            self.samples.append((filepath, label))

    def __len__(self):
        return len(self.samples)
    
    def __normalize_point_cloud(self, point_cloud):
        centroid = point_cloud.mean(axis=0)
        point_cloud -= centroid
        scale = np.max(np.linalg.norm(point_cloud, axis=1))
        point_cloud /= scale

        return point_cloud

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        data = np.loadtxt(filepath, delimiter=",").astype(np.float32) # load array of xyz and normals

        xyz, normals = data[:, 0:3], data[:, 3:6]

        # point_cloud, choices = pc_util.random_sampling(
        #     xyz, self.num_points, return_choices=True
        # )
        # normals = normals[choices] # subsample normals to match the sampled points

        if self.augment:
            point_cloud, choices = pc_util.random_sampling(
                xyz, self.num_points, return_choices=True
            )
        else:
            rng = np.random.default_rng(seed=idx)
            replace = xyz.shape[0] < self.num_points
            choices = rng.choice(xyz.shape[0], self.num_points, replace=replace)
            point_cloud = xyz[choices]
            
        normals = normals[choices]

        point_cloud = self.__normalize_point_cloud(point_cloud)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        # Augmentation inspired by PointNet++'s data augmentation for ModelNet40 classification
        if self.augment:
            # Random rotation around the Y axis
            angle = np.random.uniform() * 2 * np.pi
            cos, sin = np.cos(angle), np.sin(angle)
            # ModelNet40 uses Y-up; rotate around Y
            rot = np.array([
                [cos, 0, sin],
                [0, 1, 0],
                [-sin, 0, cos],
            ], dtype=np.float32)
            point_cloud = point_cloud @ rot.T
            normals = normals @ rot.T

            # Random scale
            point_cloud *= np.random.uniform(0.8, 1.25)

            # Random shift
            point_cloud += np.random.uniform(-0.1, 0.1, (3,))

            # Gaussian jitter
            # point_cloud += np.clip(0.01 * point_cloud.shape, -0.05, 0.05)
            point_cloud += np.random.normal(0, 0.01, size=point_cloud.shape).astype(np.float32)
            point_cloud = np.clip(point_cloud, -1.0, 1.0)
            
        if self.use_normals:
            point_cloud = np.concatenate([point_cloud, normals], axis=1)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_label"] = np.array(label).astype(np.int64)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)

        return ret_dict
