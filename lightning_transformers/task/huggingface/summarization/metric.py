# Copyright 2020 The HuggingFace Team. All rights reserved.

import re
from typing import List

import numpy as np
import torch
from filelock import FileLock
from pytorch_lightning.metrics import Metric
from rouge_score import rouge_scorer, scoring
from rouge_score.scoring import AggregateScore, Score

try:
    import nltk

    NLTK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NLTK_AVAILABLE = False

if NLTK_AVAILABLE:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


class RougeMetric(Metric):
    def __init__(
        self,
        return_precision_and_recall: bool,
        rouge_newline_sep: bool,
        use_stemmer: bool,
        rouge_keys: List[str] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),
    ):
        super().__init__()
        self.return_precision_and_recall = return_precision_and_recall
        self.rouge_newline_sep = rouge_newline_sep
        self.rouge_keys = rouge_keys
        self.use_stemmer = use_stemmer
        self.aggregator = RougeBatchAggregator()
        self.scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=self.use_stemmer)

        for key in rouge_keys:
            self.add_state(key, [])

    def update(self, pred_lns: List[str], tgt_lns: List[str]):
        for pred, tgt in zip(pred_lns, tgt_lns):
            # rougeLsum expects "\n" separated sentences within a summary
            if self.rouge_newline_sep:
                pred = add_newline_to_end_of_each_sentence(pred)
                tgt = add_newline_to_end_of_each_sentence(tgt)
            results = self.scorer.score(pred, tgt)
            for key, score in results.items():
                score = torch.tensor([score.precision, score.recall, score.fmeasure])
                getattr(self, key).append(score)

    def compute(self):
        scores = {key: getattr(self, key) for key in self.rouge_keys}
        self.aggregator.add_scores(scores)
        result = self.aggregator.aggregate()
        return format_rouge_results(result, self.return_precision_and_recall)


class RougeBatchAggregator(scoring.BootstrapAggregator):
    def aggregate(self):
        """
        Override function to wrap the final results in `Score` objects.
        This is due to the scores being replaced with a list of torch tensors.
        """
        result = {}
        for score_type, scores in self._scores.items():
            # Stack scores into a 2-d matrix of (sample, measure).
            score_matrix = np.vstack(tuple(scores))
            # Percentiles are returned as (interval, measure).
            percentiles = self._bootstrap_resample(score_matrix)
            # Extract the three intervals (low, mid, high).
            intervals = tuple((Score(*percentiles[j, :]) for j in range(3)))
            result[score_type] = AggregateScore(low=intervals[0], mid=intervals[1], high=intervals[2])
        return result

    def add_scores(self, scores):
        self._scores = scores


def format_rouge_results(result, return_precision_and_recall):
    if return_precision_and_recall:
        return extract_rouge_mid_statistics(result)  # here we return dict
    else:
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}


def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict


def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    assert NLTK_AVAILABLE, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))
