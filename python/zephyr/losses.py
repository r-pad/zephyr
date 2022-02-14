import torch
import torch.nn.functional as F


def getLoss(args):
    if args.loss == "exp_withbest":
        print("Loss on the best hypotheses")
        criterion = ExpLoss(no_best = False, cutoff = args.loss_cutoff)
    elif args.loss == "exp":
        print("No loss on the best hypotheses")
        criterion = ExpLoss(no_best = True, cutoff = args.loss_cutoff)
    elif args.loss == "inlier":
        criterion = InlierLoss()
    else:
        raise Exception("Unknown loss name:", args.loss)
    return criterion

'''
Push down the pred_y where pp_err is high
Push up pred_y where pp_err is lowest
'''
class ExpLoss():
    def __init__(self, no_best=False, cutoff = None):
        self.no_best = no_best
        self.cutoff = cutoff

    def __call__(self, pred_y, pp_err):
        pred_y = pred_y.squeeze()
        pred_y = F.softmax(pred_y, 0)
        if self.no_best:
            loss = 0
        else:
            loss = - pred_y[torch.argmin(pp_err)]
        if self.cutoff is not None:
            if self.cutoff == "log":
                pp_err = torch.log(pp_err + 1e-6)
            elif self.cutoff == "sqrt":
                pp_err = torch.sqrt(pp_err)
            else:
                self.cutoff = float(self.cutoff)
                pp_err = torch.clamp(pp_err, 0, self.cutoff)

        loss += torch.sum(pp_err * pred_y)
        return loss

'''
pp_err < thresh will be inliers
And treat it as a binary classification problem
'''
class InlierLoss():
    def __init__(self, thresh = 0.02):
        self.m = m = torch.nn.Sigmoid()
        self.criterion = torch.nn.BCELoss()
        self.thresh = thresh

    def __call__(self, pred_y, pp_err):
        y = pp_err <= self.thresh
        loss = self.criterion(self.m(pred_y), y.float().unsqueeze(-1))
        return loss
