## DenseNet with FreezeOut. 
## Adopted from Brandon Amos: https://github.com/bamos/densenet.pytorch
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

import torchvision.models as models

import sys
import math
import numpy as np

from utils import scale_fn,calc_speedup

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate,layer_index):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

        # If the layer is still being trained
        self.active=True
        
        # The index of this layer relative to the overall net
        self.layer_index=layer_index

        
    def forward(self, x):
    
        # If we're not training this layer, set to eval mode and return precomputed output
        test = DenseNet.test
        if not self.active and not test:
            self.eval()
            return x

        #while Validation , return the original output 
        if test: 
            out = self.conv1(F.relu(self.bn1(x)))
            out = self.conv2(F.relu(self.bn2(out)))
            out = torch.cat((x, out), 1)
            return out
        
        # If we're active, return a original  output.
        if self.active and not test: 
            out = self.conv1(F.relu(self.bn1(x)))
            out = self.conv2(F.relu(self.bn2(out)))
            out = torch.cat((x, out), 1)

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

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, layer_index):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
        
        # Current Layer Index
        self.layer_index = layer_index
        # If the layer is being trained or not
        self.active = True
            
    def forward(self, x):
        test = DenseNet.test
        if not self.active and not test:
            self.eval()
            return x
        if test:
            out = self.conv1(self.bn1(F.relu(x)))
            out = torch.cat((x, out), 1)
            return out
        if self.active and not test:
            out = self.conv1(self.bn1(F.relu(x)))
            out = torch.cat((x, out), 1)
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

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, layer_index):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

        # If the layer is being trained or not
        self.active = True
        
        # The layer index relative to the rest of the net
        self.layer_index = layer_index



    def forward(self, x):
        test = DenseNet.test
        # If we're not training this layer, set to eval and return the input ie precomputed output of frozen layer
        if not self.active and not test:
            self.eval()
            return x
        #While validation, return the original output 
        if test: 
           out = self.conv1(self.bn1(F.relu(x)))
           out = F.avg_pool2d(out, 2)
           return out

        # If we're active, return a original  output
        if self.active and not test:
            out = self.conv1(self.bn1(F.relu(x)))
            out = F.avg_pool2d(out, 2)
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


class DenseNet(nn.Module):
    freezeLayerIndex = -1
    test = False
    concat_output_tensor = torch.cuda.FloatTensor()
    # concat_output_label = torch.cuda.FloatTensor()
	
    concat_output_label = torch.cuda.LongTensor()	
	
	
    def __init__(self, growthRate, depth, nClasses, epochs, t_0, scale_lr=True, how_scale = 'cubic',const_time=False,reduction=0.5, bottleneck=True):
        super(DenseNet, self).__init__()
        
        self.epochs = epochs
        self.t_0 = t_0
        self.scale_lr = scale_lr
        self.how_scale = how_scale
        self.const_time = const_time

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2
            
        #Speedup calculation Not accurate as of now
        speedup = calc_speedup(growthRate,nDenseBlocks,t_0,how_scale)
        print('Estimated speedup is '+str((np.round(100*speedup)))+'%.')
        
        # Optionally scale the epochs based on the speedup so we train for
        # the same approximate wall-clock time.
        if self.const_time:
            self.epochs /= 1-speedup    
        
        
        nChannels = 2*growthRate
        #Dummy variable to keep track of freezing of First Convolution Layer
        self.first_layer = 0
		
		
#if Dataset MNIST 		
        if (nClasses == 50):
            self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1, bias=False)        
        else:
            self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
                               
#        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,
#                               bias=False)

#Store the dataset class to be used in the forward function.
        self.nClasses = nClasses
							   
        self.conv1.layer_index = 0
        self.conv1.active=True
        self.layer_index = 1
        
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels,self.layer_index)
        self.layer_index += 1

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels, self.layer_index)
        self.layer_index += 1

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)
        
        # Set bn and fc layers to active, permanently. Have them share a layer
        # index with the last conv layer.
        self.bn1.active=True
        self.fc.active=True
        self.bn1.layer_index = self.layer_index
        self.fc.layer_index = self.layer_index
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
            

            # Set the layerwise scaling and annealing parameters
            if hasattr(m,'active'):
                m.lr_ratio = scale_fn[self.how_scale](self.t_0 + (1 - self.t_0) * float(m.layer_index) / self.layer_index)
                #Changed as 1000 was a hardcoded number for Iteration wise freezing while we are doing epoch wise freezing
                m.max_j = self.epochs * m.lr_ratio
                # Optionally scale the learning rates to have the same total
                # distance traveled (modulo the gradients).
                m.lr = 1e-1 / m.lr_ratio if self.scale_lr else 1e-1
                
        # Optimizer
        self.optim = optim.SGD([{'params':m.parameters(), 'lr':m.lr, 'layer_index':m.layer_index} for m in self.modules() if hasattr(m,'active')],  
                         nesterov=True,momentum=0.9, weight_decay=1e-4)
        # Iteration Counter            
        self.j = 0  

        # A simple dummy variable that indicates we are using epoch-wise training. 
        self.lr_sched = {'itr':0}
                             
    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, self.layer_index))                
            else:
                layers.append(SingleLayer(nChannels, growthRate, self.layer_index))
            nChannels += growthRate
            self.layer_index += 1
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
                        # variable to signify First Convolution Layer is frozen
                        self.first_layer = 1
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
            DenseNet.concat_output_tensor = torch.cuda.FloatTensor()
            return train_loader

    def forward(self, x,y,test):
        DenseNet.test = test
        DenseNet.label = y
        #One Epoch before first convolution layer is frozen, pre computing the output
        if DenseNet.freezeLayerIndex == 0 and not test:
            out = self.conv1(x)
            if(DenseNet.concat_output_tensor.dim()==0):
                DenseNet.concat_output_tensor= out
                DenseNet.concat_output_label = DenseNet.label
            else:
                DenseNet.concat_output_tensor = torch.cat((DenseNet.concat_output_tensor,out),0)
                DenseNet.concat_output_label = torch.cat((DenseNet.concat_output_label, DenseNet.label),0)
            self.first_layer = 1
        #If the layer past first layer is frozen directly, get the frozen layer output
        elif self.first_layer == 0 and DenseNet.freezeLayerIndex > 0 and not test:
            out = self.conv1(x)
        elif self.first_layer == 1 and not test:
            out = x
        else:
            out = self.conv1(x)

        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
#        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))

#If MNIST dataset		
        if (self.nClasses == 50):
            out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 7))
        else:
            out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))	
			
        out = F.log_softmax(self.fc(out))
        return out
