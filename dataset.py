import json
from typing import Tuple, Dict
from torch.utils.data import Dataset


class NL2BashDataset(Dataset):
    def __init__(self, file_path: str):
        """
        Initializes a NL2BashDataset object.
        :param file_path: path to data file
        """
        super(NL2BashDataset, self).__init__()
        self.file_path = file_path
        self.data = self.parse_file()

    def parse_file(self) -> Dict:
        """
        Reads data from the dataset json file.
        :return: dataset dict
        """
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return data

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """
        Gets item by index.
        :param idx: index of item to get
        :return: the invocation and its appropriate Bash command
        """
        sample = self.data[str(idx + 1)]
        return sample["invocation"], sample["cmd"]

    def __len__(self) -> int:
        """
        Returns length of dataset.
        :return: length of dataset (number of items)
        """
        return len(self.data)
