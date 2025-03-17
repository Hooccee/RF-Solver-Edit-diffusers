from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .EditEval_v1 import EditEval_v1_dataset
from .PIE_Bench import PIE_Bench_dataset



def get_dataloader(dataset_name,default_transform=None):
    if dataset_name == 'EditEval_v1':
        return EditEval_v1_dataset(transform=default_transform)
    if dataset_name == 'PIE-Bench':
        return PIE_Bench_dataset(transform=default_transform)
