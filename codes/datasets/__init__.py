from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .EditEval_v1 import EditEval_v1_dataset
from .PIE_Bench import PIE_Bench_dataset

default_transform = transforms.Compose(
    [
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def get_dataloader(dataset_name):
    if dataset_name == 'EditEval_v1':
        return EditEval_v1_dataset(transform=default_transform)
    if dataset_name == 'PIE-Bench':
        return PIE_Bench_dataset(transform=default_transform)
