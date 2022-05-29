import pytorch_lightning as pl

from lightning_transformers.task.audio.speech_recognition import (
    SpeechRecognitionDataConfig,
    SpeechRecognitionDataModule,
    SpeechRecognitionTransformer,
)

if __name__ == "__main__":
    model = SpeechRecognitionTransformer("facebook/wav2vec2-base", ctc_loss_reduction="mean", vocab_file="vocab.json")
    dm = SpeechRecognitionDataModule(
        cfg=SpeechRecognitionDataConfig(
            batch_size=1,
            dataset_name="timit_asr",
        ),
        tokenizer=model.tokenizer,
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)