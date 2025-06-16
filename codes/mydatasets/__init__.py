from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .EditEval_v1 import EditEval_v1_dataset
from .PIE_Bench import PIE_Bench_dataset
from .PIE_Bench_shuffle import PIE_Bench_dataset_shuffle
from .EditEval_v1_shuffle import EditEval_v1_datase_shuffle
from .emu_edit_test_set import EmuEditTestSet
from .emu_edit_test_set_shuffle import EmuEditTestSet_shuffle



def get_dataloader(dataset_name,default_transform=None):
    if dataset_name == 'EditEval_v1':
        return EditEval_v1_dataset(transform=default_transform)
    if dataset_name == 'PIE-Bench':
        return PIE_Bench_dataset(transform=default_transform)
    if dataset_name == 'PIE-Bench_shuffle':
        return PIE_Bench_dataset_shuffle(transform=default_transform)
    if dataset_name == 'EditEval_v1_shuffle':
        return EditEval_v1_dataset_shuffle(transform=default_transform)
    if dataset_name == "emu_edit_test_set":
        return EmuEditTestSet(transform= default_transform)
    if dataset_name == "emu_edit_test_set_shuffle":
        return EmuEditTestSet_shuffle(transform= default_transform)
