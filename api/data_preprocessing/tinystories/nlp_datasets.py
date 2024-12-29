import torch.utils.data as data
class Dataset_ust(data.Dataset):

    def __init__(self, dataset, dataidxs=None):
        self.dataset = dataset
        self.dataidxs = dataidxs
        if self.dataidxs is not None:
            self.dataset = self.dataset.select(self.dataidxs)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample, idx

    def __len__(self):
        return len(self.dataset)