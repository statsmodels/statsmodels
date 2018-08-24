from statsmodels.tools.decorators import nottest


class Hypothesis(object):

    def __init__(self, null, alternative):
        self._null = null
        self._alternative = alternative

    @property
    def null(self):
        return self._null

    @property
    def alternative(self):
        return self._alternative

    def __str__(self):
        return ("Hypotheses:\n\t* H0: {0}\n\t* H1: {1}"
                .format(self._null, self._alternative))


class CriticalValues(object):

    def __init__(self, crit_dict):
        self._crit_dict = crit_dict

    @property
    def crit_dict(self):
        return self._crit_dict

    def __str__(self):
        items = sorted(self._crit_dict.items(),
                       key=lambda item: int(item[0].strip("%")))

        critical_values = map(lambda item: "[{0}] = {1}".format(*item),
                              items)

        return "Critical values:\n" + ", ".join(critical_values)


class Statistics(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        items = map(lambda item: "{0} = {1}".format(*item),
                    self.__dict__.items())

        return "Statistics:\n" + ", ".join(items)


@nottest
class TestResult(object):

    template = """{0}\n\n{1}\n\n{2}\n\n{3}"""

    def __init__(self, test_name, hypothesis, statistics,
                 critical_values):
        self._test_name = test_name
        self._hypothesis = hypothesis
        self._statistics = statistics
        self._critical_values = critical_values

    @property
    def hypothesis(self):
        return self._hypothesis

    @property
    def statistics(self):
        return self._statistics

    @property
    def critical_values(self):
        return self._critical_values

    def summary(self):
        return self.template.format(self._test_name,
                                    self._hypothesis,
                                    self._statistics,
                                    self._critical_values)

    def __str__(self):
        return self.summary()
