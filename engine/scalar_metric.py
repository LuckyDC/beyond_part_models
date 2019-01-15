from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class ScalarMetric(Metric):

    def update(self, value):
        self.sum_metric += value
        self.sum_inst += 1

    def reset(self):
        self.sum_inst = 0
        self.sum_metric = 0

    def compute(self):
        if self.sum_inst == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self.sum_metric / self.sum_inst
