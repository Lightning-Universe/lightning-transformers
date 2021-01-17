from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional import bleu_score


class BLEUScore(Metric):
    """
    Calculate BLEU score of machine translated text with one or more references.
    Example:
        >>> translate_corpus = ['the cat is on the mat'.split()]
        >>> reference_corpus = [['there is a cat on the mat'.split(), 'a cat is on the mat'.split()]]
        >>> metric = BLEUScore()
        >>> metric(translate_corpus, reference_corpus)
        tensor(0.7598)
    """

    def __init__(self, n_gram: int = 4, smooth: bool = False):
        """
           Args:
               n_gram: Gram value ranged from 1 to 4 (Default 4)
               smooth: Whether or not to apply smoothing – Lin et al. 2004
           """
        super().__init__()
        self.n_gram = n_gram
        self.smooth = smooth
        self.add_state('translate_corpus', [])
        self.add_state('reference_corpus', [])

    def compute(self):
        return bleu_score(
            translate_corpus=self.translate_corpus,
            reference_corpus=self.reference_corpus,
            n_gram=self.n_gram,
            smooth=self.smooth,
        )

    def update(self, translate_corpus, reference_corpus) -> None:
        """
        Actual metric computation
        Args:
            translate_corpus: An iterable of machine translated corpus
            reference_corpus: An iterable of iterables of reference corpus
        """
        self.translate_corpus += translate_corpus
        self.reference_corpus += reference_corpus
