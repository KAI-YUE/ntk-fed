from utils.utils import parse_dataset_type
import numpy as np
import pickle
import os

import torch
from torch.utils.data import Dataset

class UserDataset(Dataset):
    def __init__(self, images, labels, type_="mnist"):
        """Construct a user train_dataset and convert ndarray 
        """
        images = self._normalize(images, type_)
        labels = (labels).astype(np.int64)
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.num_samples = images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] 
        return dict(image=image, label=label)

    def _normalize(self, images, type_):
        if type_ == "mnist":
            images = images.astype(np.float32)/255
            images = (images - 0.1307)/0.3081
        elif type_ == "fmnist":
            images = images.astype(np.float32)/255
            images = (images - 0.2860)/0.3530
        elif type_ == "cifar":
            image_area = 32**2
            images = images.astype(np.float32)/255
            images[:, :image_area] = (images[:, :image_area] - 0.4914) / 0.247                              # r channel 
            images[:, image_area:2*image_area] = (images[:, image_area:2*image_area] - 0.4822) / 0.243      # g channel
            images[:, -image_area:] = (images[:, -image_area:] - 0.4465) / 0.261                            # b channel
        else: 
            images = images.astype(np.float32)/255
        
        return images

def assign_user_data(config, logger):
    """
    Load data and generate user_with_data dict given the configuration.

    Args:
        config (class):    a configuration class.
    
    Returns:
        dict: a dict contains train_data, test_data and user_with_data[userID:sampleID].
    """
    
    with open(config.train_data_dir, "rb") as fp:
        train_data = pickle.load(fp)
    
    with open(config.test_data_dir, "rb") as fp:
        test_data = pickle.load(fp)

    dataset_type = parse_dataset_type(config)
    train_data["images"] = _normalize(train_data["images"], dataset_type)
    train_onehot = np.zeros((train_data["labels"].shape[0], config.label_size), dtype=np.float32)
    train_onehot[np.arange(train_data["labels"].shape[0]), train_data["labels"]] = 1
    train_data["labels"] = train_onehot

    test_data["images"] = _normalize(test_data["images"], dataset_type)
    test_onehot = np.zeros((test_data["labels"].shape[0], config.label_size), dtype=np.float32)
    test_onehot[np.arange(test_data["labels"].shape[0]), test_data["labels"]] = 1
    test_data["labels"] = test_onehot

    if os.path.exists(config.user_with_data):
        logger.info("Non-IID data distribution")
        with open(config.user_with_data, "rb") as fp:
            user_with_data = pickle.load(fp)
    else:
        user_with_data = {}
        base = 0
        for usr_id in range(config.users):
            user_with_data[usr_id] = np.arange(base, base+config.local_batch_size)
            base += config.local_batch_size

    return dict(train_data=train_data,
                test_data=test_data,
                user_with_data=user_with_data)


def assign_user_resource(config, userID, train_dataset, user_with_data):
    """Simulate one user resource by assigning the dataset and configurations.
    """
    user_resource = {}
    batch_size = config.local_batch_size
    user_resource["lr"] = config.lr
    user_resource["device"] = config.device
    user_resource["batch_size"] = batch_size

    sampleIDs = user_with_data[userID]
    user_resource["images"] = train_dataset["images"][sampleIDs]
    user_resource["labels"] = train_dataset["labels"][sampleIDs]

    # shuffle the sampleIDs
    np.random.shuffle(user_with_data[userID])

    return user_resource

def _normalize(images, dataset_type):
    if dataset_type == "mnist":
        images = images.astype(np.float32)/255
        images = (images - 0.1307)/0.3081
    elif dataset_type == "fmnist":
        images = images.astype(np.float32)/255
        images = (images - 0.2860)/0.3530
    elif dataset_type == "cifar":
        image_area = 32**2
        images = images.astype(np.float32)/255
        images[:, :image_area] = (images[:, :image_area] - 0.4914) / 0.247                              # r channel 
        images[:, image_area:2*image_area] = (images[:, image_area:2*image_area] - 0.4822) / 0.243      # g channel
        images[:, -image_area:] = (images[:, -image_area:] - 0.4465) / 0.261                            # b channel
    else: 
        images = images.astype(np.float32)/255

    return images