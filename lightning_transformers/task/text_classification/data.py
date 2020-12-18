from lightning_transformers.core import LitTransformerDataModule


class LitTextClassificationDataModule(LitTransformerDataModule):

    def process_data(self):
        input_feature_fields = [k for k, v in self.ds['train'].features.items() if k not in ['label', 'idx']]
        self.ds = LitTextClassificationDataModule.preprocess(
            self.ds,
            self.tokenizer,
            input_feature_fields,
            self.padding,
            self.truncation,
            self.max_length
        )

        cols_to_keep = [
            x
            for x in ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'idx']
            if x in self.ds['train'].features
        ]
        self.ds.set_format("torch", columns=cols_to_keep)

    def prepare_labels(self):
        self.labels = self.ds['train'].features['labels']

    @property
    def num_classes(self):
        return self.labels.num_classes

    @property
    def data_model_kwargs(self):
        return {
            'num_classes': self.num_classes
        }

    @staticmethod
    def convert_to_features(example_batch, indices, tokenizer, input_feature_fields, padding, truncation, max_length):
        # Either encode single sentence or sentence pairs
        if len(input_feature_fields) > 1:
            texts_or_text_pairs = list(
                zip(
                    example_batch[input_feature_fields[0]],
                    example_batch[input_feature_fields[1]]
                )
            )
        else:
            texts_or_text_pairs = example_batch[input_feature_fields[0]]

        # Tokenize the text/text pairs
        features = tokenizer.tokenize(
            texts_or_text_pairs, padding=padding, truncation=truncation, max_length=max_length
        )

        # idx is unique ID we can use to link predictions to original data
        features['idx'] = indices

        return features

    @staticmethod
    def preprocess(ds, tokenizer, input_feature_fields, padding='max_length', truncation='only_first', max_length=128):
        ds = ds.map(
            LitTextClassificationDataModule.convert_to_features,
            batched=True,
            with_indices=True,
            fn_kwargs={
                'tokenizer': tokenizer,
                'input_feature_fields': input_feature_fields,
                'padding': padding,
                'truncation': truncation,
                'max_length': max_length,
            },
        )
        ds.rename_column_('label', "labels")
        return ds
