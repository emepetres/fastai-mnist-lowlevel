import pandas as pd
from fastcore.xtras import Path
from fastai.vision.all import (
    set_seed,
    Resize,
    aug_transforms,
    Normalize,
    imagenet_stats,
    DataBlock,
    ImageBlock,
    CategoryBlock,
    RandomSplitter,
    ColReader,
    vision_learner,
    resnet34,
    accuracy,
    Learner,
)

import config


def get_learner(bs: int = 64) -> Learner:
    set_seed(config.SEED)

    path = Path(config.PREPROCESSED) / "images_clean.csv"
    df = pd.read_csv(path)

    item_tfms = [Resize(224)]
    batch_tfms = [
        *aug_transforms(),
        Normalize.from_stats(*imagenet_stats),
    ]

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader(
            "image", pref=Path(config.INPUTS) / "images_compressed", suff=".jpg"
        ),
        splitter=RandomSplitter(),
        get_y=ColReader("label"),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )

    dls = dblock.dataloaders(df)  # , bs=bs)

    return vision_learner(dls, resnet34, metrics=accuracy)
