import torc
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable

from giana import GIANA



root = '/local-scratch/saeedI/CVC-VideoClinicDBtrain/Seqs/1'
dset = GIANA(root)
data_loader = data.DataLoader(dset, batch_size=1)

for batch_index, (inputs, targets) in enumerate(data_loader):
    inputs = Variable(inputs)
    print np.shape(inputs.data.numpy())
print len(dset)
