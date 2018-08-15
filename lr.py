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

class OneCycle(object):
    def __init__(self, optimizer, epochs, lr_from, lr_to, momentum_from, momentum_to, anneal_pct, anneal_rate):
        self.optimizer = optimizer

        # Total number of train epochs
        cycle_len = int(epochs * (1 - anneal_pct))

        self.lr = itertools.chain.from_iterable([
            linear_decay(lr_from, lr_to, cycle_len / 2),
            linear_decay(lr_to, lr_from, cycle_len / 2),
            div_decay(lr_from, anneal_rate, epochs - cycle_len)
            ])

        self.momentum = itertools.chain.from_iterable([
            linear_decay(momentum_from, momentum_to, cycle_len / 2),
            linear_decay(momentum_to, momentum_from, cycle_len / 2),
            [momentum_from for i in range(cycle_len, epochs)]
            ])

    def step(self):
        try:
            lr = next(self.lr)
            momentum = next(self.momentum)
            tqdm.write("Change LR to {}, momentum to {}".format(lr, momentum))

            for param_group in self.optimizer.param_groups:
                tqdm.write("Previous LR {}, momentum {}".format(param_group["lr"], param_group["momentum"]))
                param_group['lr'] = lr
                param_group['momentum'] = momentum

        except e:
            pass
