import pytorch_lightning as pl
from transformers import AutoTokenizer

from lightning_transformers.task.nlp.question_answering import QuestionAnsweringTransformer, SquadDataModule

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")
    model = QuestionAnsweringTransformer(pretrained_model_name_or_path="bert-base-uncased")
    dm = SquadDataModule(
        batch_size=1,
        dataset_config_name="plain_text",
        max_length=384,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0,
        doc_stride=128,
        n_best_size=20,
        max_answer_length=30,
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)

    trainer.fit(model, dm)
