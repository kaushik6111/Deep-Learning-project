from pathlib import Path
import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def resolve_data_dir(custom_path=None):
    root = Path(custom_path) if custom_path else Path(kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia'))
    candidates = [root, root / 'chest_xray', root / 'chest_xray' / 'chest_xray']
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in ('train', 'val', 'test')):
            return candidate
    raise FileNotFoundError(f'Cannot find train/val/test under {root}')

def get_loaders(data_dir, batch_size=32):
    common = [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    augment = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(10), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.1, contrast=0.1), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    sets = {s: datasets.ImageFolder(Path(data_dir) / s, augment if s == 'train' else transforms.Compose(common)) for s in ('train', 'val', 'test')}
    assert sets['train'].classes == ['NORMAL', 'PNEUMONIA'], f'Unexpected class order: {sets["train"].classes}'
    loaders = {s: DataLoader(ds, batch_size=batch_size, shuffle=(s == 'train'), num_workers=0, pin_memory=False) for s, ds in sets.items()}
    return loaders, sets
