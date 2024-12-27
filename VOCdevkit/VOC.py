import os

import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
from PIL import Image


def load_voc_data(path, classes, split="train"):
    """
    loading the PASCAL VOC2012 data

    args:
        path: root directory of Vof OC2012
        classes: classes to load
        split: train or val
    """
    base_path = os.path.join(path, "VOC2012")
    # VOC2012/ImageSets/Main, txt that includes all the info
    image_set_path = os.path.join(base_path, "ImageSets", "Main")
    # print(image_set_path)

    # VOC2012/JPEGImages, images that contain the classes
    image_dir = os.path.join(base_path, "JPEGImages")
    # print(image_path)

    # dataset = {image_id: image_path}
    dataset = {}

    # labels = {image_id: [0,0,0,0,0]}
    # array keeps track of classes present in the image
    # ["bicycle", "motorbike", "person", "cat", "train"]
    labels = {}

    for cls_idx, cls in enumerate(classes):
        file_name = f"{cls}_{split}.txt"
        file_path = os.path.join(image_set_path, file_name)
        # print("current class...", cls)

        # open the current file
        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            image_id, label = line.strip().split()
            # print(image_id, label)
            # 0 mean difficult to tell if class is present
            # -1 not object of class in the image
            # 1 object of class in the image
            if label == "0":
                continue

            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            if os.path.exists(image_path):
                if image_id not in dataset:
                    labels[image_id] = np.zeros(len(classes))
                    dataset[image_id] = image_path

                # 0 not present, 1 present
                labels[image_id][cls_idx] = 1 if label == "1" else 0
    dataset = list(dataset.values())
    labels = np.array(list(labels.values()))
    # print(f"length of {split} dataset...", len(dataset))
    # print(f"length of {split} labels...", len(labels))
    # print(labels)
    return dataset, labels


file_dir = os.getcwd()

classes = ["bicycle", "motorbike", "person", "cat", "train"]

train_images, train_labels = load_voc_data(file_dir, classes, "train")
val_images, val_labels = load_voc_data(file_dir, classes, "val")


def EDA(labels, classes, split):
    """
    perform data analysis
    """
    class_count = labels.sum(axis=0)
    print(f"class distribution in {split}:")
    for cls, count in zip(classes, class_count):
        print(f"{cls:10}: {int(count):4d} Images")

    plt.figure(figsize=(10, 5))
    plt.bar(classes, class_count)
    plt.title(f"class distribution for {split}")
    plt.ylabel("no. of Images")
    plt.tight_layout()
    plt.show()


EDA(train_labels, classes, "train")
EDA(val_labels, classes, "val")
