import random
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed.
    :param seed: random seed
    :return:
    """
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False