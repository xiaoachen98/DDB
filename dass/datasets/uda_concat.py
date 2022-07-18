from mmseg.datasets import ConcatDataset
from .builder import DATASETS


@DATASETS.register_module()
class UDAConcatDataset(ConcatDataset):
    def __init__(self, datasets, separate_eval=True):
        try:
            super(ConcatDataset, self).__init__(datasets)
        except NotImplementedError as e:
            print(e)
            print(
                "Since our program does not use the special authentication method of cityscapes, we ignore this warning"
            )
        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = datasets[0].PALETTE
        self.separate_eval = separate_eval
        assert separate_eval in [True, False], (
            f"separate_eval can only be True or False," f"but get {separate_eval}"
        )
