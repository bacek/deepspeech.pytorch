import os
from tqdm import tqdm


class Observer(object):
    '''
    Train Observer base class.
    '''
    def on_epoch_start(self, model, epoch): pass
    def on_epoch_end(self, model, epoch, avg_loss, wer, cer): pass
    def on_batch_start(self, model, epoch, batch_no): pass
    def on_batch_end(self, model, epoch, batch_no, avg_loss): pass


def to_np(x):
    return x.data.cpu().numpy()


class TensorboardWriter(Observer):
    def __init__(self, id, log_dir, log_params):
        os.makedirs(log_dir, exist_ok=True)
        from tensorboardX import SummaryWriter

        self.id = id
        self.log_params = log_params
        self.tensorboard_writer = SummaryWriter(log_dir)

    def on_epoch_end(self, model, epoch, avg_loss, wer, cer):
        tqdm.write("Updating tensorboard for epoch {}".format((epoch + 1)))
        values = {
            'Avg Train Loss': avg_loss,
            'Avg WER': wer,
            'Avg CER': cer
        }
        self.tensorboard_writer.add_scalars(self.id, values, epoch + 1)
        if self.log_params:
            for tag, value in model.named_parameters():
                tqdm.write("Param {}".format(tag))
                tag = tag.replace('.', '/')
                self.tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
                if value.grad is not None:
                    self.tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)
