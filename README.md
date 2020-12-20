# lightning-transformers

### Current API
```bash
# Train bert-base-cased on SQuAD using question answering task provided by huggingface
python train.py +task/question_answering=huggingface +dataset/question_answering=squad

# Train bert-base-cased on CARER emotion dataset using text classification task provided by huggingface
python train.py +task/text_classification=huggingface +dataset/text_classification=emotion

# (An example, not real) Swap to fairseq model that is compatible with text classification task provided by huggingface
python train.py +task/text_classification=huggingface +model=fairseq +dataset/text_classification=emotion
```