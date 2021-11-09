import os
import pickle
import logging
import datetime
import numpy as np

import torch

from fedlearning.model import init_weights
from fedlearning import nn_registry

def init_logger(config):
    """Initialize a logger object. 
    """
    log_level = logging.INFO    
    logger = logging.getLogger()
    logger.setLevel(log_level)

    fh = logging.FileHandler(config.log_file)
    fh.setLevel(log_level)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("-"*80)

    return logger

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_record(config, model):
    record = {}
    # number of trainable parameters
    record["num_parameters"] = count_parameters(model)

    # put some config info into record
    record["batch_size"] = config.local_batch_size
    record["lr"] = config.lr
    record["taus"] = []

    # initialize data record 
    record["testing_accuracy"] = []
    record["loss"] = []

    return record

def save_record(config, record):
    current_path = os.path.dirname(__file__)
    current_time = datetime.datetime.now()
    current_time_str = datetime.datetime.strftime(current_time ,'%H_%M')
    file_name = config.record_dir.format(current_time_str)
    with open(os.path.join(current_path, file_name), "wb") as fp:
        pickle.dump(record, fp)


def parse_model(config):
    if config.model in nn_registry.keys():
        return nn_registry[config.model]

    if "cifar" in config.train_data_dir:
        return nn_registry["cifar_mlp"]
    elif "fmnist" in config.train_data_dir:
        return nn_registry["fmnist_mlp"]
    else:
        return nn_registry["mnist_mlp"]

def parse_dataset_type(config):
    if "fmnist" in config.train_data_dir:
        type_ = "fmnist"
    elif "mnist" in config.train_data_dir:
        type_ = "mnist"
    elif "cifar" in config.train_data_dir:
        type_ = "cifar"
    
    return type_

def init_model(config, logger):
    # initialize the model
    sample_size = config.datapoint_size[0] * config.datapoint_size[1] * config.channels
    full_model = nn_registry[config.model](in_dims=sample_size, in_channels=config.channels, out_dims=config.label_size)
    full_model.apply(init_weights)

    if os.path.exists(config.full_weight_dir):
        logger.info("--- Load pre-trained full precision model. ---")
        state_dict = torch.load(config.full_weight_dir)
        full_model.load_state_dict(state_dict)

    full_model.to(config.device)

    return full_model