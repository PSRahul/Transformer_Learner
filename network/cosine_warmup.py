import torch.optim as optim
import numpy as np


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iterations):
        super().__init__(optimizer)
        self.warmup = warmup
        self.max_iterations = max_iterations

    def get_lr(self):
        learning_rate_factor = self.get_learning_rate_factor(
            epoch=self.last_epoch)

        return [base_lr*learning_rate_factor for base_lr in self.base_lrs]

    def get_learning_rate_factor(self, epoch):
        learning_rate_factor = 0.5 * \
            (1 + np.cos(np.pi*epoch/self.max_iterations))
        if epoch <= self.warmup:
            learning_rate_factor *= epoch/self.warmup*1.0
        return learning_rate_factor
