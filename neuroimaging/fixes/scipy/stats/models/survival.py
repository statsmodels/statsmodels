import numpy as np

class SurvivalTime(object):
    def __init__(self, time, delta):
        self.time, self.delta = time, delta

    def atrisk(self, time):
        raise NotImplementedError

class RightCensored(SurvivalTime):

    def atrisk(self, time):
        return np.less_equal.outer(time, self.time)

class LeftCensored(SurvivalTime):

    def atrisk(self, time):
        return np.greater_equal.outer(time, self.time)
