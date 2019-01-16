from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class WarmupMultiStepScheduler(LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_begin_lr=0.0,
                 warmup_epoch=0, mode='gradual', last_epoch=-1):

        assert isinstance(milestones, list) and len(milestones) >= 1
        for i, step_ in enumerate(milestones):
            if i != 0 and milestones[i] <= milestones[i - 1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if step_ < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if gamma > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        if mode not in ["constant", "gradual"]:
            raise ValueError("Mode must be \"gradual\" or \"constant\"")
        if warmup_epoch >= milestones[0]:
            raise ValueError("Warmup_epoch must be smaller than first milestone.")

        self.gamma = gamma
        self.milestones = milestones
        self.warmup_epoch = warmup_epoch
        self.warmup_begin_lr = warmup_begin_lr

        super(WarmupMultiStepScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch:
            return [self.warmup_begin_lr + (base_lr - self.warmup_begin_lr) / self.warmup_epoch * self.last_epoch
                    for base_lr in self.base_lrs]
        else:
            return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                    for base_lr in self.base_lrs]
