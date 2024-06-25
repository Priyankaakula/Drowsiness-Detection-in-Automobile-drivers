import json
import os

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm


IMAGENET_RES = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_WEIGHTS = [1/38967, 1/8869, 1/5495]
ds_labels = {"alert": 0, "microsleep": 1, "yawning": 2}
label_ds = {v:k for k, v in ds_labels.items()}
#ROOT_DIR = "/d/hpc/projects/FRI/DL/mm1706/" # "../../data"
ROOT_DIR = "C:/GWAR/dataset"

annot_files = ["classification_frames/annotations_train.json",
            "classification_frames/annotations_val.json",
            "classification_frames/annotations_test.json"]


class DriverStateDataset(Dataset):
    """Driver state (classification) dataset"""

    def __init__(self, data_dir, img_paths, labels, label_dict, transform=None):
        self.data_dir = data_dir
        self.img_paths = img_paths
        self.labels = labels
        self.label_dict = label_dict
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_paths[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        driver_state = self.label_dict[self.labels[idx]["driver_state"]]
        img = self.transform(image=img)["image"] if self.transform else img
        img = ToTensorV2()(image=img)['image']
        driver_state = torch.tensor(driver_state)
        return img, driver_state


def get_dataloaders(
    data_dir, annot_files: list, class_weights: list = None,
    transforms: list = [None, None, None], batch_size: int = 32
):
    """Get train, validation and test dataloader."""
    assert len(transforms) == 3, "transforms must be a list of length 3 (train trans, val trans, test trans)"
    assert len(annot_files) == 3, "annot_files must be a list of length 3 (train annot, val annot, test annot)"

    # create datasets
    datasets = []
    for idx in range(3):
        annots = json.load(open(os.path.join(data_dir, annot_files[idx])))
        img_paths = list(annots.keys())
        labels = list(annots.values())
        dataset = DriverStateDataset(data_dir, img_paths, labels, ds_labels, transform=transforms[idx])
        datasets.append(dataset)

    # compute sample weights
    desc = "Computing sample weights"
    if class_weights is not None:
        sample_weights = []
        for _, label in tqdm(datasets[0], total=len(datasets[0]), desc=desc):
            sample_weights.append(class_weights[label])
        # oversample the minority classes in the training set.
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(datasets[0]), replacement=True)
        train_dataloder = DataLoader(datasets[0], batch_size, sampler=sampler)
    else:
        train_dataloder = DataLoader(datasets[0], batch_size)
    val_dataloader = DataLoader(datasets[1], batch_size)
    test_dataloader = DataLoader(datasets[2], batch_size)
    return {"train": train_dataloder, "val": val_dataloader, "test": test_dataloader}, ds_labels, label_ds


def get_dataloader(
    data_dir, annot_file, transform, ds_labels, batch_size: int = 32
):
    """Get data-loader."""
    annots = json.load(open(os.path.join(data_dir, annot_file)))
    img_paths = list(annots.keys())
    labels = list(annots.values())
    dataset = DriverStateDataset(data_dir, img_paths, labels, ds_labels, transform)

    return DataLoader(dataset, batch_size)


def load_driver_data(params):
    """Load driver state data."""
    print(f"loading driver data...")
    dataloaders, ds_labels, labels_ds = get_dataloaders(
        data_dir=ROOT_DIR,
        annot_files=annot_files,
        class_weights=CLASS_WEIGHTS,
        transforms=[get_train_transform(), get_test_transform(), get_test_transform()],
        batch_size=params["batch_size"]
    )
    return dataloaders, ds_labels, labels_ds


def get_train_transform():
    """Get the train transform."""
    transform = A.Compose([
        A.OneOf([
            A.Rotate(limit=20, p=0.70, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.30)],
            p=0.75),
        A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.75),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Resize(width=IMAGENET_RES[0], height=IMAGENET_RES[1]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def get_test_transform():
    """Get the test transform."""
    transform = A.Compose([
        A.Resize(width=IMAGENET_RES[0], height=IMAGENET_RES[1]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def get_basic_transform():
    """Get the basic transform (only resizes the image)."""
    return A.Compose([A.Resize(width=224, height=224)])


if __name__ == "__main__":
    params = json.load(open(os.path.join(os.getcwd(), "hyper_params.json")))
    dataloaders, ds_labels, labels_ds = load_driver_data(params)
