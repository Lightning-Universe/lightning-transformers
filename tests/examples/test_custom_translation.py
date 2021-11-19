# import pytorch_lightning as pl
# from transformers import AutoTokenizer
#
# from examples.custom_translation.dataset import MyTranslationDataModule
# from examples.custom_translation.model import MyTranslationTransformer
# from lightning_transformers.core.nlp import HFBackboneConfig
# from lightning_transformers.task.nlp.translation.config import TranslationDataConfig
#
#
# def test_example(hf_cache_path):
#     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="patrickvonplaten/t5-tiny-random")
#     model = MyTranslationTransformer(
#         backbone=HFBackboneConfig(pretrained_model_name_or_path="patrickvonplaten/t5-tiny-random")
#     )
#     dm = MyTranslationDataModule(
#         cfg=TranslationDataConfig(
#             batch_size=1,
#             dataset_name="wmt16",
#             dataset_config_name="ro-en",
#             source_language="en",
#             target_language="ro",
#             cache_dir=hf_cache_path,
#             limit_train_samples=16,
#             limit_val_samples=16,
#             limit_test_samples=16,
#             max_source_length=32,
#             max_target_length=32,
#             preprocessing_num_workers=1,
#         ),
#         tokenizer=tokenizer,
#     )
#     trainer = pl.Trainer(fast_dev_run=True)
#
#     trainer.fit(model, dm)
