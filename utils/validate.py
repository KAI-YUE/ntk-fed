import numpy as np

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# My libraries
from utils.utils import parse_dataset_type
from fedlearning.dataset import UserDataset

def validate_and_log(model, dataset, config, record, logger, verbose=True):

    with torch.no_grad():
        model.eval()
        # validate the model and log test accuracy
        loss = train_loss(model, dataset["train_data"], loss_type=config.loss, device=config.device)
        test_acc = test_accuracy(model, dataset["test_data"], device=config.device)
        
        if verbose:
            record["loss"].append(loss)
            record["testing_accuracy"].append(test_acc)

            logger.info("Test accuracy {:.4f}".format(test_acc))
            logger.info("Train loss {:.4f}".format(loss))
            logger.info("")

        model.train()

        return test_acc, loss

def test_accuracy(model, test_dataset, device="cuda"):
    with torch.no_grad():
        model.eval()
        out = model(torch.from_numpy(test_dataset["images"]).to(device))
        pred_labels = torch.argmax(out, dim=-1)
        true_labels = torch.argmax(torch.from_numpy(test_dataset["labels"]).to(device), dim=-1)
        accuracy = torch.sum(pred_labels == true_labels)/pred_labels.shape[0]
    return accuracy.item()

def accuracy_with_output(output, labels):
    pred_labels = torch.argmax(output, dim=-1)

    if type(labels) == torch.tensor:
        labels = torch.argmax(torch.from_numpy(labels).to(output), dim=-1)
    else:
        labels = torch.argmax(labels, dim=-1)

    accuracy = torch.sum(pred_labels == labels)/pred_labels.shape[0]
    return accuracy.item()

def loss_with_output(output, labels, loss_type):
    if loss_type == "ce":
        label_size = labels.shape[1]
        criterion = lambda fx, y: -label_size*torch.mean(torch.log(torch.softmax(fx, dim=1))*y)
    elif loss_type == "mse":
        criterion =  lambda yhat, y: 0.5*torch.mean((yhat - y)**2) 
    else:
        raise NotImplementedError

    if type(labels) == torch.tensor:
        loss = criterion(output, torch.from_numpy(labels).to(output))
    else:
        loss = criterion(output, labels)

    return loss.item()   


def train_loss(model, train_dataset, loss_type, device="cuda"):
    with torch.no_grad():
        # criterion = nn.CrossEntropyLoss()
        if loss_type == "ce":
            label_size = train_dataset["labels"].shape[1]
            criterion = lambda fx, y: -label_size*torch.mean(torch.log(torch.softmax(fx, dim=1))*y)
        elif loss_type == "mse":
            criterion =  lambda yhat, y: 0.5*torch.mean((yhat - y)**2) 
        else:
            raise NotImplementedError

        out = model(torch.from_numpy(train_dataset["images"]).to(device))
        loss = criterion(out, torch.from_numpy(train_dataset["labels"]).to(device))

    return loss.item()