# lightning-transformers

### PRINCIPAL CMD

```python
python train.py --task={{TASK}} --model={{MODEL}} --dataset={{DATASET}} ...
```

### SUPPORTED COMBINAISONS

| `{{TASK}}` | `{{DATASET}}` | `{{MODEL}}` | CMD_LINE | WORKING      |
| ------------- | ------------- | --------------------------- | ---------------------------- | ---------------------------- |
| question_answering  | bert-base-uncased  | squad | ``python train.py --task question_answering --model bert-base-uncased --dataset squad`` | `True`  | 