"""Dataset fusion utility for combining multiple datasets with sampling ratios."""

import random
from torch.utils.data import Dataset


class FusionDataset(Dataset):
    """Fuses multiple datasets with specified sampling ratios.

    Example:
        sirs = SIRSDataset(...)
        syn  = SynthesisDataset(...)
        fused = FusionDataset([sirs, syn], [0.6, 0.4], size=10000)
    """

    def __init__(self, datasets, fusion_ratios=None, size=None):
        self.datasets = datasets
        self.size = size or sum(len(d) for d in datasets)
        self.fusion_ratios = fusion_ratios or [1. / len(datasets)] * len(datasets)

        print(f'[FusionDataset] Fusing {len(datasets)} datasets:')
        for i, (dataset, ratio) in enumerate(zip(datasets, self.fusion_ratios)):
            print(f'  - Dataset {i}: {len(dataset)} samples, ratio={ratio:.2f}')
        print(f'  Total size: {self.size}')

    def reset(self):
        """Reset all datasets that support reset()."""
        for dataset in self.datasets:
            if hasattr(dataset, 'reset'):
                dataset.reset()

    def __getitem__(self, index):
        residual = 1.0
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio / residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index % len(dataset)]
            residual -= ratio

    def __len__(self):
        return self.size
