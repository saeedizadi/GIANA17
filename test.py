import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

from giana import GIANA



root = '/local-scratch/saeedI/CVC-VideoClinicDBtrain/Seqs/1'
dset = GIANA(root)
data_loader = data.DataLoader(dset, batch_size=1)

for batch_index, (inputs, targets) in enumerate(data_loader):
    inputs = Variable(inputs)
    a = inputs.data.numpy()
    imgplot = plt.imshow(a[0,:,:,:])
    plt.show()

print len(dset)
