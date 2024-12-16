class ResultsStore:
    def __str__(self):
        return getattr(self, '_str', self.__class__.__name__)


__all__ = ["ResultsStore"]
