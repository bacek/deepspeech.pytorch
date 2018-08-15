import itertools
from tqdm import tqdm

def linear_decay(frm, to, steps):
    '''
    Generate linear values between fmr and to for number of steps.
    '''
    pct = (to-frm)/(steps-1)
    step = 0
    while step < steps:
        val = frm + pct * step
        step += 1
        yield val

def div_decay(frm, anneal, steps):
    val = frm
    step = 0
    while step < steps:
        yield val
        val /= anneal
        step += 1

class SGDSchedule(object):
    '''
    LR and Momentum updater.

    Params:
    * lr Learning Rate iterable
    * momentum Optional Momentum iterable
    '''
    def __init__(self, optimizer, lr, momentum):
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum

    def step(self):
        try:
            lr = next(self.lr) if self.lr else None
        except:
            lr = None

        try:
            momentum = next(self.momentum) if self.momentum else None
        except:
            momentum = None

        tqdm.write("Change LR to {}, momentum to {}".format(lr, momentum))

        for param_group in self.optimizer.param_groups:
            tqdm.write("Previous LR {}, momentum {}".format(param_group["lr"], param_group["momentum"]))
            if lr is not None: param_group['lr'] = lr
            if momentum is not None: param_group['momentum'] = momentum

class OneCycle(SGDSchedule):
    def __init__(self, optimizer, epochs, lr_from, lr_to, momentum_from, momentum_to, anneal_pct, anneal_rate):
        # Total number of train epochs
        cycle_len = int(epochs * (1 - anneal_pct))

        lr = itertools.chain.from_iterable([
            linear_decay(lr_from, lr_to, cycle_len / 2),
            linear_decay(lr_to, lr_from, cycle_len / 2),
            div_decay(lr_from, anneal_rate, epochs - cycle_len)
            ])

        momentum = itertools.chain.from_iterable([
            linear_decay(momentum_from, momentum_to, cycle_len / 2),
            linear_decay(momentum_to, momentum_from, cycle_len / 2),
            [momentum_from for i in range(cycle_len, epochs)]
            ])

        super().__init__(optimizer, lr, momentum, None)

class Anneal(SGDSchedule):
    def __init__(self, optimizer, epochs, lr, learning_anneal):
        lr = div_decay(lr, learning_anneal, epochs)
        super().__init__(optimizer, lr, None)
