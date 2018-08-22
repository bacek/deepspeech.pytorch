import logging
import itertools

def linear_decay(frm, to, steps):
    '''
    Generate linear values between fmr and to for number of steps.
    '''
    pct = (to-frm)/steps
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
    def __init__(self, logger, optimizer, lr, momentum):
        self.logger = logger
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

        self.logger.debug("Change LR to {}, momentum to {}".format(lr, momentum))

        for param_group in self.optimizer.param_groups:
            self.logger.debug("Previous LR {}, momentum {}".format(param_group["lr"], param_group["momentum"]))
            if lr is not None: param_group['lr'] = lr
            if momentum is not None: param_group['momentum'] = momentum

class OneCycle(SGDSchedule):
    def __init__(self, optimizer, epochs, lr_from, lr_to, momentum_from, momentum_to, anneal_pct, anneal_rate):
        # Total number of train epochs
        cycle_len = int(epochs * (1 - anneal_pct) / 2)

        lr = itertools.chain.from_iterable([
            linear_decay(lr_from, lr_to, cycle_len),
            linear_decay(lr_to, lr_from, cycle_len),
            div_decay(lr_from, anneal_rate, epochs - 2 * cycle_len)
            ])

        momentum = itertools.chain.from_iterable([
            linear_decay(momentum_from, momentum_to, cycle_len),
            linear_decay(momentum_to, momentum_from, cycle_len),
            [momentum_from for i in range(cycle_len, epochs)]
            ])

        super().__init__(logging.getLogger('lr.OneCycle'), optimizer, lr, momentum)

        self.logger.info("lr: {}->{} m: {}->{}".format(lr_from, lr_to, momentum_from, momentum_to))

class Anneal(SGDSchedule):
    def __init__(self, optimizer, epochs, lr, learning_anneal):
        lr = div_decay(lr, learning_anneal, epochs)
        super().__init__(logging.getLogger('lr.Anneal'), optimizer, lr, None)
