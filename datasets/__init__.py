# Copyright (c) Facebook, Inc. and its affiliates.
from .scannet import ScannetDetectionDataset, ScannetDatasetConfig
from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig
from .modelnet40 import ModelNet40ClassificationDataset, ModelNet40DatasetConfig


DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
    "modelnet40": [ModelNet40ClassificationDataset, ModelNet40DatasetConfig],
}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()

    if args.dataset_name == "modelnet40":
        dataset_dict = {
            "train": dataset_builder(
                dataset_config,
                split_set="train",
                root_dir=args.dataset_root_dir,
                use_normals=args.use_normals,
                augment=True,
            ),
            "test": dataset_builder(
                dataset_config,
                split_set="test",
                root_dir=args.dataset_root_dir,
                use_normals=args.use_normals,
                augment=False,
            ),
        }
    else:
        dataset_dict = {
            "train": dataset_builder(
                dataset_config,
                split_set="train",
                root_dir=args.dataset_root_dir,
                meta_data_dir=args.meta_data_dir,
                use_color=args.use_color,
                augment=True,
            ),
            "test": dataset_builder(
                dataset_config,
                split_set="val",
                root_dir=args.dataset_root_dir,
                use_color=args.use_color,
                augment=False,
            ),
        }
    return dataset_dict, dataset_config
    