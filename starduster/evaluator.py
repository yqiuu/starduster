import numpy as np
import torch


__all__ = ['Evaluator', 'validate']


class Evaluator:
    def __init__(self, model, opt, loss, labels=None, scheduler=None, n_out=1):
        self.model = model
        self.opt = opt
        self.loss = loss
        self.labels = ("loss",) if labels is None else labels
        self.scheduler = scheduler
        self.n_out = n_out


    def loss_func(self, *args):
        if len(args) == 1:
            x, = args
            return self.loss(self.model(x), x)
        else:
            x = args[:-self.n_out]
            x = x[0] if len(x) == 1 else x
            y = args[-self.n_out:]
            y = y[0] if len(y) == 1 else y
            return self.loss(self.model(x), y)


    def call(self, data, backward=True):
        values = [None]*len(data)
        for i_b, d_b in enumerate(data):
            if backward:
                l_b = self.loss_func(*d_b)
                if isinstance(l_b, torch.Tensor):
                    l_b = [l_b]
                l_b[0].backward()
                self.opt.step()
                self.opt.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
            else:
                self.model.eval()
                l_b = self.loss_func(*d_b)
                self.model.train()
                if isinstance(l_b, torch.Tensor):
                    l_b = [l_b]
            values[i_b] = l_b
        values = torch.mean(torch.tensor(values), dim=0).detach().tolist()
        lr = self.opt.param_groups[0]['lr']
        return values, lr


    def train(self, dl_train, dl_valid, n_epochs=100):
        history = {'epoch':[0]*n_epochs}
        for name in self.labels:
            history[name] = [0]*n_epochs
            history[f"{name}_val"] = [0]*n_epochs

        history_train = [None]*n_epochs
        history_valid = [None]*n_epochs
        history_lr = [None]*n_epochs
        for i_e in range(n_epochs):
            history_train[i_e], history_lr[i_e] \
                = self.call(dl_train, backward=True)
            history_valid[i_e], _ = self.call(dl_valid, backward=False)

            msg = ["{}(val)={:.2e}({:.2e})".format(*values) for values in \
                zip(self.labels, history_train[i_e], history_valid[i_e])]
            msg = "\repoch={}, ".format(i_e + 1) + ", ".join(msg)
            print(msg, end="")
        print()

        history = {}
        history['epoch'] = np.arange(1, n_epochs + 1)
        history['lr'] = np.asarray(history_lr)
        for name, values in zip(self.labels, zip(*history_train)):
            history[name] = np.asarray(values)
        for name, values in zip(self.labels, zip(*history_valid)):
            history[f"{name}_val"] = np.asarray(values)

        return history


def validate(model, metric, x, y_true, numpy=True):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        err = metric(y_pred, y_true)
        
    if numpy:
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        err = err.cpu().numpy()
        
    return err, y_pred, y_true

