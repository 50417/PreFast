# VGG net stolen from the TorchVision package.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import numpy as np

from utils import scale_fn
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
class MaxPool(nn.Module):
    def __init__(self, n_in, n_out, layer_index):
        super(MaxPool, self).__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer_index = layer_index
        self.active = True

    def forward(self,x):
        test = DenseNet.test
        if not self.active and not test:
            self.eval()
            return x
        if test:
            out = self.maxPool(x)
            return out
        # If we're active, return a original  output.
        if self.active and not test:
            out = self.maxPool(x) 
             #Start saving the output once we know we are one step before the  freezing point
            if(self.layer_index == DenseNet.freezeLayerIndex):
                if(DenseNet.concat_output_tensor.dim()==0):
                    DenseNet.concat_output_tensor= out
                    DenseNet.concat_output_label = DenseNet.label
                else:
                    DenseNet.concat_output_tensor = torch.cat((DenseNet.concat_output_tensor,out),0)
                    DenseNet.concat_output_label = torch.cat((DenseNet.concat_output_label, DenseNet.label),0)
            return out
        else:
            return x        

class Layer(nn.Module):
    def __init__(self, n_in, n_out, layer_index):
        super(Layer, self).__init__()
        
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_out)
        
        self.layer_index = layer_index
        # If the layer is being trained or not
        self.active = True

            
    def forward(self, x):
        test = DenseNet.test
        if not self.active and not test:
            self.eval()
            return x
        #while Validation, return the original output
        if test:
            out = F.relu(self.bn1(self.conv1(x)))
            return out
        # If we're active, return a original  output.
        if self.active and not test: 
            out = F.relu(self.bn1(self.conv1(x)))
             #Start saving the output once we know we are one step before the  freezing point
            if(self.layer_index == DenseNet.freezeLayerIndex):
                if(DenseNet.concat_output_tensor.dim()==0):
                    DenseNet.concat_output_tensor= out
                    DenseNet.concat_output_label = DenseNet.label
                else:
                    DenseNet.concat_output_tensor = torch.cat((DenseNet.concat_output_tensor,out),0)
                    DenseNet.concat_output_label = torch.cat((DenseNet.concat_output_label, DenseNet.label),0)
            return out
        else:
            return x        

# Using the VGG values provided by Sergey Zagoryuko in http://torch.ch/blog/2015/07/30/cifar.html
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # 'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    }
# It's VGG but we call it DenseNet for compatibility with the training loop.
# I'll fix it later.
# GrowthRate and Depth are ignored.
class DenseNet(nn.Module):
    freezeLayerIndex = -1
    test = False
    counter_layer = {}
    concat_output_tensor = torch.FloatTensor()
    concat_output_label = torch.FloatTensor()
    def __init__(self,growthRate, depth, nClasses, epochs, t_0, scale_lr=True, how_scale = 'cubic',const_time=False, cfg=cfg['E'],batch_norm=True):
        super(DenseNet, self).__init__()
        
        self.epochs = epochs
        self.t_0 = t_0
        self.scale_lr = scale_lr
        self.how_scale = how_scale
        self.const_time = const_time
        
        self.layer_index = 0
        self.features = self.make_layers(cfg,batch_norm)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.Linear(512, nClasses),
        )
        self.classifier.layer_index = self.layer_index
        self.classifier.active = True
        self._initialize_weights()
        
        # Optimizer
        self.optim = optim.SGD([{'params':m.parameters(), 'lr':m.lr, 'layer_index':m.layer_index} for m in self.modules() if hasattr(m,'active')],  
                         nesterov=True,momentum=0.9, weight_decay=1e-4)
        # Iteration Counter            
        self.j = 0  

        # A simple dummy variable that indicates we are using an iteration-wise
        # annealing scheme as opposed to epoch-wise. 
        self.lr_sched = {'itr':0}
        
    def forward(self, x, y,test):
        DenseNet.test = test
        DenseNet.label = y
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.classifier(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            # Set the layerwise scaling and annealing parameters
            if hasattr(m,'active'):
                m.lr_ratio = scale_fn[self.how_scale](self.t_0 + (1 - self.t_0) * float(m.layer_index) / self.layer_index)
                 #Iteration per epoch
                #self.itr_epoch = train_sz//batch_sz
                
                #Changed as 1000 was a hardcoded number for Iteration wise freezing while we are doing epoch wise freezing
                m.max_j = self.epochs  * m.lr_ratio
                
                # Optionally scale the learning rates to have the same total
                # distance traveled (modulo the gradients).
                m.lr = 0.1 / m.lr_ratio if self.scale_lr else 0.1
        

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [MaxPool( in_channels,v,self.layer_index)]
                self.layer_index += 1
            else:
                # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                # if batch_norm:
                layers += [Layer( in_channels,v,self.layer_index)]
                self.layer_index += 1
                # else:
                    # layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def update_lr(self,batch_size):
    
                # Loop over all modules
        for m in self.modules():
            # If a module is active:
            if hasattr(m,'active') and m.active:
                # If we are one epoch away from  layer's freezing point, get the layerIndex to start saving output.
                if self.j > m.max_j-1:
                    DenseNet.freezeLayerIndex = m.layer_index
                    if self.j > m.max_j:
                        # If we've passed this layer's freezing point, deactivate it.
                        m.active = False
                        # Also make sure we remove all this layer from the optimizer
                        for i,group in enumerate(self.optim.param_groups):
                            if group['layer_index']==m.layer_index:
                                self.optim.param_groups.remove(group)
                    #Update the learning rate  if not at freezing point
                    else: 
                        for i,group in enumerate(self.optim.param_groups):
                            if group['layer_index']==m.layer_index:
                                self.optim.param_groups[i]['lr'] = (0.05/m.lr_ratio)*(1+np.cos(np.pi*self.j/m.max_j))\
                                                          if self.scale_lr else 0.05 * (1+np.cos(np.pi*self.j/m.max_j))
                
                # If not, update the LR
                else:
                    for i,group in enumerate(self.optim.param_groups):
                        if group['layer_index']==m.layer_index:
                            self.optim.param_groups[i]['lr'] = (0.05/m.lr_ratio)*(1+np.cos(np.pi*self.j/m.max_j))\
                                                              if self.scale_lr else 0.05 * (1+np.cos(np.pi*self.j/m.max_j))
        
        # Update the iteration counter
        self.j += 1
        if DenseNet.concat_output_tensor.dim() != 0:
            train_data = DenseNet.concat_output_tensor.data
            train_label = DenseNet.concat_output_label
            train_set = data_utils.TensorDataset(train_data, train_label)
            train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True)
            #Reinitializing the used Tensor Memory
            DenseNet.concat_output_tensor = torch.FloatTensor()
            return train_loader

