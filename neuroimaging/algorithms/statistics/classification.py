"""
TODO
"""
__docformat__ = 'restructuredtext'

from nipy.fixes.scipy.stats.models.model import Model

class Classifier(Model):
    """
    TODO
    """
    def learn(self, **keywords):
        """
        :Parameters:
            keywords : ``dict``
                TODO

        :Returns: ``None``
        """
        self.fit(**keywords)

