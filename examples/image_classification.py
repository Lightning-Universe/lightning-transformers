import pytorch_lightning as pl
from transformers import AutoFeatureExtractor

from lightning_transformers.task.vision.image_classification import (
    ImageClassificationDataConfig,
    ImageClassificationDataModule,
    ImageClassificationTransformer,
)

feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path="nateraw/vit-base-beans")
dm = ImageClassificationDataModule(
    cfg=ImageClassificationDataConfig(batch_size=8, dataset_name="beans", num_workers=8),
    feature_extractor=feature_extractor,
)
model = ImageClassificationTransformer(
    pretrained_model_name_or_path="nateraw/vit-base-beans", num_labels=dm.num_classes
)

trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=5)
trainer.fit(model, dm)
