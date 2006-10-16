from scipy.sandbox.models.model import Model

class Classifier(Model):
    def learn(self, **keywords):
        self.fit(**keywords)

