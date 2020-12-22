from pytorch_lightning import _logger as log 
from lightning_transformers.tasks.multiple_choice.core.data import (
    DataProcessor,
    InputExample
)

class SynonymProcessor(DataProcessor):
    """Processor for the Synonym data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "mctrain.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "mchp.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))

        return self._create_examples(self._read_csv(os.path.join(data_dir, "mctest.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""

        examples = [
            InputExample(
                example_id=line[0],
                question="",  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[1], line[1], line[1], line[1], line[1]],
                endings=[line[2], line[3], line[4], line[5], line[6]],
                label=line[7],
            )
            for line in lines  # we skip the line with the column names
        ]

        return examples