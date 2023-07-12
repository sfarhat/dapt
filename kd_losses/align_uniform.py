import torch
import torch.nn as nn
import wandb

class AlignUniform(nn.Module):

    def __init__(self, embed_s, embed_t, align_alpha, unif_t, align_w, unif_w, logging=True):
        super(AlignUniform, self).__init__()
        self.embed_s = embed_s
        self.embed_t = embed_t
        self.align_alpha = align_alpha
        self.unif_t = unif_t
        self.align_w = align_w
        self.unif_w = unif_w
        self.logging = logging

    def align_loss(self, x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniform_loss(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def forward(self, feat_s, feat_t):

        feat_s = self.embed_s(feat_s)
        feat_t = self.embed_t(feat_t)

        align_loss_val = self.align_loss(feat_s, feat_t, alpha=self.align_alpha)
        unif_loss_val = (self.uniform_loss(feat_s, t=self.unif_t) + self.uniform_loss(feat_t, t=self.unif_t)) / 2

        loss = align_loss_val * self.align_w + unif_loss_val * self.unif_w

        if self.logging:
            wandb.log({"Align Loss": align_loss_val.item(),
                    "Uniform Loss": unif_loss_val.item()})

        return loss
