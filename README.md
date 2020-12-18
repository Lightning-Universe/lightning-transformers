# lightning-transformers

```bash
# Training pre-trained bert-base-cased on SQuAD
python train.py +model/question_answering=huggingface +dataset/question_answering=emotion

# Training pre-trained bert-base-cased on CARER emotion dataset
python train.py +model/text_classification=huggingface +dataset/text_classification=emotion
```