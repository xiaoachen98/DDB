from mmseg.datasets import CityscapesDataset, CustomDataset
from .builder import DATASETS

@DATASETS.register_module()
class MapillaryDataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        super(MapillaryDataset, self).__init__(
            img_suffix=".jpg", seg_map_suffix=".png", **kwargs
        )
