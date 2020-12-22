
from pytorch_lightning import _logger as log 
from lightning_transformers.tasks.multiple_choice.core.data import (
    DataProcessor,
    InputExample
)


class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        log.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        log.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        log.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        # There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "1234":
                return int(truth) - 1
            else:
                log.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[
                            options[0]["para"].replace("_", ""),
                            options[1]["para"].replace("_", ""),
                            options[2]["para"].replace("_", ""),
                            options[3]["para"].replace("_", ""),
                        ],
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth,
                    )
                )

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        log.info("len examples: %s}", str(len(examples)))
        log.info("Three choices: %s", str(three_choice))
        log.info("Five choices: %s", str(five_choice))
        log.info("Other choices: %s", str(other_choices))
        log.info("four choices: %s", str(four_choice))

        return examples