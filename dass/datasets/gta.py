from mmseg.datasets import CityscapesDataset, CustomDataset
from .builder import DATASETS


@DATASETS.register_module()
class GTADataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get("split") in [None, "train"]
        if "split" in kwargs:
            kwargs.pop("split")
        super(GTADataset, self).__init__(
            img_suffix=".png", seg_map_suffix="_labelTrainIds.png", split=None, **kwargs
        )
