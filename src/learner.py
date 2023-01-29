import pandas as pd
from fastcore.xtras import Path
from fastai.vision.all import (
    DataLoaders,
    set_seed,
    untar_data,
    get_image_files,
    URLs,
    GrandparentSplitter,
    Datasets,
    PILImageBW,
    parent_label,
    Categorize,
    CropPad,
    RandomCrop,
    ToTensor,
    IntToFloatTensor,
    Normalize,
    resnet18,
    nn,
    Normalize,
    Learner,
    accuracy,
)

import config


def get_dataloaders() -> DataLoaders:
    set_seed(config.SEED)
    path = untar_data(URLs.MNIST)
    items = get_image_files(path)

    splitter = GrandparentSplitter(
        train_name="training",
        valid_name="testing",
    )

    dsrc = Datasets(
        items,
        tfms=[
            [PILImageBW.create],  # path -> x transforms
            [parent_label, Categorize],  # path -> y transforms
        ],
        # Categorize is label encoding
        splits=splitter(items),
    )

    item_tfms = [CropPad(34), RandomCrop(size=28), ToTensor()]
    batch_tfms = [IntToFloatTensor(), Normalize()]

    # When outside the DataBlock API, item_tfms and batch_tfms will always be
    # referenced as after_item and after_batch, including inside the dataloader itself.
    dls = dsrc.dataloaders(bs=128, after_item=item_tfms, after_batch=batch_tfms)

    return dls


def get_learner(bs: int = 64) -> Learner:

    dls = get_dataloaders()

    model = resnet18(num_classes=dls.c)
    # adapt resnet to one channel inputs
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    )
    model.cuda()

    return Learner(dls, model, metrics=[accuracy])
