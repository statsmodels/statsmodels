from neuroimaging.algorithms.statistics import Model

class Classifier(Model):
    def learn(self, **keywords): self.fit(**keywords)

