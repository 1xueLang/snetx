import torch
import torch.nn as nn

import snetx
from snetx.dataset import vision
from snetx.snn import algorithm as snnalgo
from snetx.snn import neuron
# from snetx.cuend import neuron
from snetx.models import svggnet

class MNISTNet(nn.Module):
    def __init__(self, T):
        super().__init__()
        
        self.T = T
        self.encod = snnalgo.DirectEncoder(T)
        
        self.conv = nn.Sequential(
            snnalgo.Tosnn(nn.Conv2d(1, 32, 3, 1, 1)),
            snnalgo.Tosnn(nn.BatchNorm2d(32)),
            neuron.LIF(),
            
            snnalgo.Tosnn(nn.Conv2d(32, 32, 3, 2, 1)),
            snnalgo.Tosnn(nn.BatchNorm2d(32)),
            neuron.LIF(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 14 * 14, 800),
            neuron.LIF(),
            nn.Linear(800, 10)
        )
    
    def forward(self, x):
        h = self.encod(x)
        return self.classifier(
            self.conv(h).flatten(2)
        )

if __name__ == '__main__':
    data_dir = '../../dataset'
    T = 5
    batch_size1 = 100
    batch_size2 = 100
    lr = 1e-3
    device = torch.device('cuda')
    snetx.utils.seed_all()
    
    ds1, ds2 = vision.mnist_dataset(data_dir, batch_size1, batch_size2)
    net = MNISTNet(T).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = snnalgo.TET(torch.nn.CrossEntropyLoss())
    # criterion = snnalgo.TET(torch.nn.MSELoss())
    
    for e in range(100):
        
        correct = 0.0
        total = 0.0
        
        for i, (x, y) in enumerate(ds1):
            x = x.to(device)
            y = y.to(device)
            # y = torch.nn.functional.one_hot(y, 10).to(x)
            
            optimizer.zero_grad()
            
            out = net(x)
            loss = criterion(y, out)
            loss.backward()
            
            optimizer.step()
            
            total += out.shape[0]
            correct += out.mean(dim=1).argmax(dim=1).eq(y).sum()
            
            if (1 + i) % 10 == 0:
                print(loss.item()) 
                print(correct / total)
