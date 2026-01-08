import os
import sys
import shutil
from sklearn.model_selection import train_test_split
from torchvision import datasets


sys.path = list(dict.fromkeys(sys.path))  # reset path first
# paths relative to project root
PROJECT_ROOT = os.path.abspath("..")
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

ROOT_DATA = os.path.join(PROJECT_ROOT, "data")
RAW_PATH = os.path.join(ROOT_DATA, "raw")
SPLIT_PATH = os.path.join(ROOT_DATA, "splits")

# EuroSAT specific subpath created by torchvision
EUROSAT_IMAGE_DIR = os.path.join(RAW_PATH, "eurosat", "2750")


# for checking if the Raw Data exists and not empty
def dir_exists_and_not_empty(path):
    return os.path.isdir(path) and any(os.scandir(path))

# for checking if the splits directories exists and not empty
def valid_splits_exist(split_path):
    if not os.path.isdir(split_path):
        return False
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(split_path, split)
        if not dir_exists_and_not_empty(split_dir):
            return False
    return True

def setup_eurosat():

    # check if Raw Data exists
    if not dir_exists_and_not_empty(EUROSAT_IMAGE_DIR):
        print(f"Raw EuroSAT data not found or empty at {EUROSAT_IMAGE_DIR}. Downloading...")
        os.makedirs(RAW_PATH, exist_ok=True)
        datasets.EuroSAT(root=RAW_PATH, download=True)
    else:
        print("Raw EuroSAT data already exists and is non-empty. Skipping download.")

    # check if Splits already exist
    if valid_splits_exist(SPLIT_PATH):
        print(f"Valid train/val/test splits already exist at {SPLIT_PATH}. Skipping split logic.")
        return

    if os.path.exists(SPLIT_PATH):
        print("Incomplete or invalid splits found. Rebuilding splits...")
        shutil.rmtree(SPLIT_PATH)

    print("Creating stratified splits...")

    dataset = datasets.ImageFolder(EUROSAT_IMAGE_DIR)
    indices = list(range(len(dataset)))
    labels = dataset.targets

    # stratified split (80/10/10)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=3
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        stratify=[labels[i] for i in temp_idx],
        random_state=3
    )

    # move images according to split indices
    def build_split(idx_list, split_name):
        for i in idx_list:
            img_path, target = dataset.samples[i]
            cls_name = dataset.classes[target]
            dest_dir = os.path.join(SPLIT_PATH, split_name, cls_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(img_path, dest_dir)

    for name, idxs in zip(["train", "val", "test"], [train_idx, val_idx, test_idx]):
        print(f"Processing {name} split...")
        build_split(idxs, name)

    print("Data setup complete.")


if __name__ == "__main__":
    setup_eurosat()
