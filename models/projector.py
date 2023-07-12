import torch
import torch.nn as nn

class Embed(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, projector):
        super(Embed, self).__init__()

        if projector == 'simclr':
            # Projector used in SimCLR
            self.projector = nn.Sequential(
                nn.Linear(in_dim, in_dim, bias=False),
                nn.BatchNorm1d(in_dim), 
                nn.ReLU(),
                nn.Linear(in_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim, affine=False), 
                _Normalize()
                )
        elif projector == 'simclrv2':
            # Projector used in SimCLRv2
            self.projector = nn.Sequential(
                nn.Linear(in_dim, in_dim, bias=False),
                nn.BatchNorm1d(in_dim), 
                nn.ReLU(),
                nn.Linear(in_dim, in_dim, bias=False),
                nn.BatchNorm1d(in_dim), 
                nn.ReLU(),
                nn.Linear(in_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim, affine=False), 
                _Normalize()
                )
        elif projector == 'mocov2':
            # Projector used in MoCoV2
            # Technically, MoCoV2 has hidden_dim == in_dim
            self.projector = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
                _Normalize()
                )
        elif projector == 'linear':
            self.projector = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                _Normalize()
                )
        else:
            raise NotImplementedError(projector)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        return x

class _Normalize(nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=-1)
