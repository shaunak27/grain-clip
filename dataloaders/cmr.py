import glob
import os
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
import pandas as pd
class Flickr30k(VisionDataset):
    """`Flickr30k Entities <https://bryanplummer.com/Flickr30kEntities/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)

        # Read annotations and store in a dict
        self.annotations = defaultdict(list)
        fh = pd.read_csv(self.ann_file)
        for i in range(len(fh)):
            caption = fh['raw'].iloc[i].strip('[').strip(']').replace('"','').split(',')
            img_id = fh['filename'].iloc[i]
            splity = fh['split'].iloc[i]
            if splity=="test":
                self.annotations[img_id] = caption

        self.ids = list(sorted(self.annotations.keys()))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]

        # Image
        filename = os.path.join(self.root, img_id)
        img = Image.open(filename).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


    def __len__(self) -> int:
        return len(self.ids)