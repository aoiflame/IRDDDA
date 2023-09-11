import numpy as np
import torch

class SaveBest:
    def __init__(self, verbose=False, delta=0, ptpath=''):
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.val_score_min = 0
        self.delta = delta
        self.ptpath = ptpath
        self.renew = -1

    def __call__(self, val_score, model, prefix, epoch, x_emb, y_emb):

        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, prefix, x_emb, y_emb)
            self.renew = epoch
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_score, model, prefix, x_emb, y_emb)
            self.renew = epoch
        else:
            pass

    def save_checkpoint(self, val_score, model, prefix, x_emb, y_emb):
        if self.verbose:
            print(f'Validation score increased ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), '{}/{}_checkpoint.pt'.format(self.ptpath, prefix))
        np.save('{}/oversample/x_oversample_{}.npy'.format(self.ptpath, prefix), x_emb.cpu().detach())
        np.save('{}/oversample/y_oversample_{}.npy'.format(self.ptpath, prefix), y_emb.cpu().detach())
        self.val_score_min = val_score
