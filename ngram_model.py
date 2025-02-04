import re
from collections import Counter, defaultdict
import math


class NGramModel:

    def __init__(self, n: int = 3, level: str = 'token') -> None:
        """Initialize the NGramModel.

        Args:
            n (int, optional): The size of the n-grams. Defaults to 3.
            level (str, optional): The level of n-grams, can be 'token' or 'char'. Defaults to 'token'.
        """
        self.n = n
        self.level = level
        self.ngrams: defaultdict[tuple[str, ...],
                                 Counter[str]] = defaultdict(Counter)
        self.total_ngrams: Counter[tuple[str, ...]] = Counter()
        self.token_count: Counter[str] = Counter()
        self.vocabulary: set[str] = set()

    def from_text(self, text: str) -> None:
        """Build the model from a given text."""
        tokens = self._tokenize(text)
        self.from_tokenized(tokens)

    def from_tokenized(self, tokens: list[str]) -> None:
        """Count n-grams from a list of tokens."""
        tokens = [token.lower() for token in tokens]
        tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        self.vocabulary.update(tokens)
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            self.ngrams[ngram[:-1]][ngram[-1]] += 1
            self.total_ngrams[ngram[:-1]] += 1
            self.token_count[ngram[-1]] += 1

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization via regex."""
        text = text.lower()
        if self.level == 'char':
            tokens = list(text.replace(" ", ""))
        else:
            tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def get_prior(self, token: str) -> float:
        """Get the prior probability of a token."""
        token = token.lower()
        if token in self.token_count:
            return self.token_count[token] / sum(self.token_count.values())
        else:
            # Handle out-of-vocabulary tokens
            return 1 / (sum(self.token_count.values()) + len(self.vocabulary))

    def get_posterior(self, context: tuple[str, ...], token: str) -> float:
        """Get the posterior probability of a token given a context."""
        context = tuple(token.lower() for token in context)
        token = token.lower()
        if context in self.total_ngrams:
            # Laplace smoothing
            return (self.ngrams[context][token] + 1) / (self.total_ngrams[context] + len(self.vocabulary))
        else:
            return self.get_prior(token)

    def get_next_token(self, context: tuple[str, ...]) -> str:
        """Get the next token with the highest posterior probability given a context."""
        context = tuple(token.lower() for token in context)
        max_prob = 0
        max_token = None
        for token in self.vocabulary:
            prob = self.get_posterior(context, token)
            if prob > max_prob:
                max_prob = prob
                max_token = token
        return max_token if max_token is not None else '</s>'


def compute_log_probability(PWhisper_Y_given_X_T, P_NGM_Y_given_T, lambda_NGM=0.5):
    log_PWhisper = math.log(PWhisper_Y_given_X_T)
    log_P_NGM = math.log(P_NGM_Y_given_T)

    LP = 1 / (1 + lambda_NGM) * (log_PWhisper + lambda_NGM * log_P_NGM)

    return LP
