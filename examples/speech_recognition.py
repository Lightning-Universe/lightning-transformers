import pytorch_lightning as pl
from transformers import Wav2Vec2CTCTokenizer

from lightning_transformers.task.nlp.speech_recognition import (
    SpeechRecognitionConfig,
    SpeechRecognitionDataConfig,
    SpeechRecognitionTransformer,
    SpeechRecognitionDataModule,
)

if __name__ == "__main__":
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/wav2vec2-base")
    model = SpeechRecognitionTransformer(
        pretrained_model_name_or_path="facebook/wav2vec2-base",
        cfg=SpeechRecognitionConfig(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False
        ),
    )
    dm = SpeechRecognitionDataModule(
        cfg=SpeechRecognitionDataConfig(
            dataset_name="common_voice",
            subset="en",
            sampling_rate=16_000,
            max_length=5
        ),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)
