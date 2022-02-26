from matplotlib import transforms
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)
    
class Encoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Encoder, self).__init__()
        c, h, w = shape
        ww = ((w-8)//2 - 4)//2
        hh = ((h-8)//2 - 4)//2
        self.encode = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=28*28, out_features=400),
                      nn.BatchNorm1d(400),
                      nn.LeakyReLU(),
                                    MLP([400, 128])
                                   )

    def forward(self, x, y = None):
        x = self.encode(x)
        return x
        if (y is None):
            return self.calc_mean(x), self.calc_logvar(x)
        else:
            return self.calc_mean(torch.cat((x, y), dim=1)), self.calc_logvar(torch.cat((x, y), dim=1))

class Decoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Decoder, self).__init__()
        c, w, h = shape
        self.shape = shape
        self.decode = nn.Sequential(MLP([nhid+ncond, 64, 128, 256, c*w*h], last_activation = False), nn.Sigmoid())
        self.invTrans = transforms.Compose([
                                    transforms.Normalize((0.1307,), (0.3081,))
                        ])
    def forward(self, z, y = None):
        c, w, h = self.shape
        if (y is None):
            return self.invTrans(self.decode(z).view(-1, c, w, h))
        else:
            return self.invTrans(self.decode(torch.cat((z, y), dim=1)).view(-1, c, w, h))

class Solver(nn.Module):
  def __init__(self, vae, nhid = 16):
    super().__init__()
    self.input_dim = nhid
    self.vae = vae

  def forward(self, x):
    self.vae.encoder(x)

class VAE(nn.Module):
    def __init__(self, shape, nhid = 16, n_classes=10):
        super(VAE, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(shape, nhid)
        self.calc_mean = MLP([128, nhid], last_activation = False)
        self.calc_logvar = MLP([128, nhid], last_activation = False)
        self.classification = MLP([128, n_classes], last_activation = False)
        self.decoder = Decoder(shape, nhid)
        
    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma
    
    # Orginial forward of VAE. We modify this to tie it in with Avalanche plugin syntax
    #def forward(self, x):
    #    mean, logvar = self.encoder(x)
    #    z = self.sampling(mean, logvar)
    #    return self.decoder(z), mean, logvar
    def forward(self, x):
        return self.encoder(x)
    
    def generate(self, batch_size = None):
        z = torch.randn((batch_size, self.dim)).to(device) if batch_size else torch.randn((1, self.dim)).to(device)
        res = self.decoder(z)
        if not batch_size:
            res = res.squeeze(0)
        return res

class cVAE(nn.Module):
    def __init__(self, shape, nclass, nhid = 16, ncond = 16):
        super(cVAE, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(shape, nhid, ncond = ncond)
        self.calc_mean = MLP([128+ncond, nhid], last_activation = False)
        self.calc_logvar = MLP([128+ncond, nhid], last_activation = False)
        self.decoder = Decoder(shape, nhid, ncond = ncond)
        self.label_embedding = nn.Embedding(nclass, ncond)
        
    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma
    
    def forward(self, x, y):
        y = self.label_embedding(y)
        mean, logvar = self.encoder(x, y)
        z = self.sampling(mean, logvar)
        return self.decoder(z, y), mean, logvar
    
    def generate(self, class_idx):
        if (type(class_idx) is int):
            class_idx = torch.tensor(class_idx)
        class_idx = class_idx.to(device)
        if (len(class_idx.shape) == 0):
            batch_size = None
            class_idx = class_idx.unsqueeze(0)
            z = torch.randn((1, self.dim)).to(device)
        else:
            batch_size = class_idx.shape[0]
            z = torch.randn((batch_size, self.dim)).to(device) 
        y = self.label_embedding(class_idx)
        res = self.decoder(z, y)
        if not batch_size:
            res = res.squeeze(0)
        return res

# Loss functions    
BCE_loss = nn.BCELoss(reduction = "sum")
MSE_loss = nn.MSELoss(reduction = "sum")
CE_loss = nn.CrossEntropyLoss()
def VAE_loss(X, X_hat, mean, logvar):
    reconstruction_loss = MSE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    return reconstruction_loss + KL_divergence