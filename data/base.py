import numpy as np
import _pickle as cPickle
from torch.utils.data import Dataset

def multi_label_to_onehot(labels, num_classes):
    onehot = np.zeros(num_classes)
    onehot[labels] = 1
    return onehot

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]  
    

class BaseTrain(BaseDataset):
    def __init__(self, pkl_path, num_classes):
        with open(pkl_path, 'rb') as f:
            self.data = cPickle.load(f)
        self.num_classes = num_classes
        self.subjects = list(self.data.keys())

    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, i):
        latent = self.data[self.subjects[i]]['latent']
        label = self.data[self.subjects[i]]['label']
        onehot_label = multi_label_to_onehot(label, self.num_classes)

        return latent, onehot_label


class BaseTest(BaseDataset):
    def __init__(self, pkl_path, num_classes):
        with open(pkl_path, 'rb') as f:
            self.data = cPickle.load(f)
        self.num_classes = num_classes
        self.subjects = list(self.data.keys())

    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, i):
        latent = self.data[self.subjects[i]]['latent']
        label = self.data[self.subjects[i]]['label']
        onehot_label = multi_label_to_onehot(label, self.num_classes)

        return latent, onehot_label