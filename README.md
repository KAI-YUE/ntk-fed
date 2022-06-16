## NTK Empowered Federated Learning
![](https://img.shields.io/badge/Python-3-blue) ![](https://img.shields.io/badge/Pytorch-1.9.0-blue) ![](https://img.shields.io/badge/arxiv-2110.03681-red)

In this [paper](https://arxiv.org/abs/2110.03681), we propose a novel federated learning paradigm empowered by the neural tangent kernel (NTK) framework. The paradigm addresses the challenge of statistical heterogeneity by transmitting update data that are more expressive than those of the traditional FL paradigms. Compared to traditional algorithms such as FedAvg, NTK-FL has a more centralized training flavor by transmitting more expressive updates. This repository contains the simulations for the federated learning paradigm.

<br />

### Prerequisites

```bash
pip3 install -r requirements.txt
```
The datasets have been preprocessed under `data` directory.

<br />

### Example


Run the example with fashion mnist dataset: 
```bash
python3 train_fmnist.py
```
The script will load the configuration file `config_fmnist.yaml` and data matrices under `data/fmnist/`. 

You can change the dataset, for example, to EMNIST, by modifying the `config.yaml` to 
```
train_data_dir: data/emnist/digits/train.dat
test_data_dir:  data/emnist/digits/test.dat
```

<br />

### Change the Degree of Heterogeneity

The `user_with_data` files predefine the [Dirichlet non-IID partitions [HQB19]](https://arxiv.org/abs/1909.06335)  with different degrees of heterogeneity.  If you want to generate different partitions, you can use the following code snippets:
```python
"""
For each client, sample q~Dir(alpha, p).
"""
alpha = 0.5
num_users = 300
num_classes = 10
num_datapoints = 60000

samples_per_user = int(y_train[:num_datapoints].shape[0]/num_users)
samples_per_class = int(y_train[:num_datapoints].shape[0]/num_classes)
user_dataidx_map = {}

# if balance_trick:
idxs_ascending_labels = np.argsort(y_train[:num_datapoints])
labels_idx_map = np.zeros((num_classes, samples_per_class))
for i in range(num_classes):
    labels_idx_map[i] = idxs_ascending_labels[i*samples_per_class:(i+1)*samples_per_class]
    np.random.shuffle(labels_idx_map[i])
    
for user_id in range(num_users):
    current_user_dataidx = []
    proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
    histogram = samples_per_user*proportions
    histogram = histogram.astype(np.int)
    
    for i in range(num_classes):
        current_user_dataidx.append(labels_idx_map[i][:histogram[i]])
        np.random.shuffle(labels_idx_map[i])
        
    user_dataidx_map[user_id] = np.hstack(current_user_dataidx).astype(np.int).flatten()
``` 

<br />

### Citation
```
@inproceedings{yue2022neural,
  title={Neural Tangent Kernel Empowered Federated Learning},
  author={Yue, Kai and Jin, Richeng and Pilgrim, Ryan and Wong, Chau-Wai and Baron, Dror and Dai, Huaiyu},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```