# SYSTEMS AND METHODS FOR PHYSICS-INFORMED NEURAL NETWORKS FOR ROBOT MAPPING
# Copyright © 2025 Purdue University.
# Developed by Yuchen Liu, Ruiqi Ni and CORAL Lab.
# Purdue Research Foundation Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.
import cProfile, pstats, io
import matplotlib
import numpy as np
import math
import random
import time
import os
from datetime import datetime, timedelta

import torch
from torch.nn import Linear
from torch import Tensor
from torch.autograd import Variable, grad
from torchvision import transforms


import matplotlib
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import pickle

from timeit import default_timer as timer

import igib_runner
import gridmap
from dataprocessing import isdf_sample, data_pc_validate, transform 
import json 
import cv2 
from load_njsdf.inference import load_sdf_2d_model
torch.backends.cudnn.benchmark = True
from chance_constrained_planning.rollout import rollout_optimized
from chance_constrained_planning.optimizer import solve_step
EXPLORATION = 1 
READ_FROM_COOKED_DATA = 2
def find_deviation_region(traj, opt, threshold=0.0005):
    if traj is None or opt is None:
        return None

    traj = np.asarray(traj)
    opt  = np.asarray(opt)

    n = min(len(traj), len(opt))

    diff = np.linalg.norm(traj[:n, :2] - opt[:n, :2], axis=1)

    idx = np.where(diff > threshold)[0]

    if len(idx) == 0:
        return None

    return idx
def compute_zoom_bounds(points, padding=0.1):
    xmin = np.min(points[:,0]) - padding
    xmax = np.max(points[:,0]) + padding
    ymin = np.min(points[:,1]) - padding
    ymax = np.max(points[:,1]) + padding
    return xmin, xmax, ymin, ymax

def plot_traj_difference(folder, epoch, traj_list, optimized_traj_list, model):

    traj = np.asarray(traj_list)
    opt  = np.asarray(optimized_traj_list)

    n = min(len(traj), len(opt))
    traj = traj[:n]
    opt  = opt[:n]

    idx = find_deviation_region(traj, opt, threshold=0.002)

    if idx is None:
        return

    # create velocity field grid (same as plot())
    limit = 1
    xmin = [-0.5, -0.5]
    xmax = [0.5, 0.5]
    spacing = limit/80.0

    X, Y = np.meshgrid(
        np.arange(xmin[0], xmax[0], spacing),
        np.arange(xmin[1], xmax[1], spacing)
    )

    Xsrc = traj[0]

    XP = np.zeros((len(X.flatten()), 2*model.dim))
    XP[:,:model.dim] = Xsrc
    XP[:,model.dim:] = Xsrc

    XP[:,model.dim+0] = X.flatten()
    XP[:,model.dim+1] = Y.flatten()

    XP = torch.tensor(XP, dtype=torch.float32).to(model.Params['Device'])

    ss = model.Speed(XP)
    V  = ss.detach().cpu().numpy().reshape(X.shape)

    pts = np.vstack((traj[idx], opt[idx]))

    xmin, xmax, ymin, ymax = compute_zoom_bounds(pts, padding=0.1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    quad1 = ax.pcolormesh(X, Y, V, vmin=0, vmax=1)

    ax.plot(traj[:,0], traj[:,1],
            color='red',
            linewidth=1,
            label='Nominal trajectory')

    ax.plot(opt[:,0], opt[:,1],
            color='blue',
            linewidth=1,
            label='Optimized trajectory')

    ax.scatter(traj[idx,0], traj[idx,1], color='red', s=20)
    ax.scatter(opt[idx,0], opt[idx,1], color='blue', s=20)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.legend()

    plt.colorbar(quad1, ax=ax)

    plt.title("Local Optimization Difference")

    plt.savefig(folder + f"/traj_diff_zoom_{epoch}.png",
                bbox_inches='tight')

    plt.close(fig)
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def sigmoid(input):
 
    return torch.sigmoid(10*input)

class Sigmoid(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return sigmoid(input) 

class DSigmoid(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return 10*sigmoid(input)*(1-sigmoid(input)) 

def sigmoid_out(input):
 
    return torch.sigmoid(0.1*input)

class Sigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return sigmoid_out(input) 

class DSigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return 0.1*sigmoid_out(input)*(1-sigmoid_out(input)) 

class DDSigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return 0.01*sigmoid_out(input)*(1-sigmoid_out(input))*(1-2*sigmoid_out(input))


class NN(torch.nn.Module):
    
    def __init__(self, device, dim ,B):#10
        super(NN, self).__init__()
        self.dim = dim

        h_size = 128 #512,256
        #input_size = 128
        #self.T=2

        self.B = B.T.to(device)
        print(B.shape)
        input_size = B.shape[0]
        #decoder

        self.scale = 10

        self.act = torch.nn.Softplus(beta=self.scale)#ELU,CELU

        #self.env_act = torch.nn.Sigmoid()#ELU
        #self.ddact = torch.nn.Sigmoid()-torch.nn.Sigmoid()*torch.nn.Sigmoid()
        self.actout = Sigmoid_out()#ELU,CELU

        #self.env_act = torch.nn.Sigmoid()#ELU

        self.nl1=3
        self.nl2=2

        self.encoder = torch.nn.ModuleList()
        self.encoder1 = torch.nn.ModuleList()
        #self.encoder.append(Linear(self.dim,h_size))
        
        self.encoder.append(Linear(2*input_size,h_size))
        self.encoder1.append(Linear(2*input_size,h_size))
        
        for i in range(self.nl1-1):
            self.encoder.append(Linear(h_size, h_size)) 
            self.encoder1.append(Linear(h_size, h_size)) 
        
        self.encoder.append(Linear(h_size, h_size)) 

        self.generator = torch.nn.ModuleList()
        self.generator1 = torch.nn.ModuleList()
        for i in range(self.nl2):
            self.generator.append(Linear(h_size, h_size)) 
            self.generator1.append(Linear(2*h_size, 2*h_size)) 
        
        self.generator.append(Linear(h_size,h_size))
        self.generator.append(Linear(h_size,1))
    
    def init_weights(self, m):
        
        if type(m) == torch.nn.Linear:
            stdv = (1. / math.sqrt(m.weight.size(1))/1.)*2
            #stdv = np.sqrt(6 / 64.) / self.T
            m.weight.data.uniform_(-stdv, stdv)
            m.bias.data.uniform_(-stdv, stdv)
    
    def input_mapping(self, x):
        w = 2.*np.pi*self.B
        x_proj = x @ w
        #x_proj = (2.*np.pi*x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)    #  2*len(B)

    def out(self, coords):
        '''
        returns: x starts with being input to the NN, then gets encodes, and finally becomes the output tau,
        XP is [start_coords, goal_coords]
        '''
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        size = coords.shape[0]
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        
        x = torch.vstack((x0,x1))

        # x = x.unsqueeze(1)
        
        x = self.input_mapping(x)
        x  = torch.sin(self.encoder[0](x))
        for ii in range(1,self.nl1):
            #i0 = x
            x_tmp = x
            x  = torch.sin(self.encoder[ii](x))
            # x  = self.act(self.encoder1[ii](x) + x_tmp) 

        
        x = self.encoder[-1](x)

        x0 = torch.sin(x[:size,...])
        x1 = torch.sin(x[size:,...])

        # xx = torch.cat((x0, x1), dim=1)
        
        # x_0 = torch.logsumexp(self.scale*xx, 1)/self.scale
        # x_1 = -torch.logsumexp(-self.scale*xx, 1)/self.scale

        # x = torch.cat((x_0, x_1),1)
        x = (x0-x1)**2#torch.sqrt((x0-x1)**2 + 0.01)
        # x = torch.abs(x0-x1)
        for ii in range(self.nl2):
            x_tmp = x
            x = self.act(self.generator[ii](x)) 
            # x = self.act(self.generator1[ii](x) + x_tmp) 
            # x = torch.sin(self.generator[ii](x))
            # x = torch.sin(self.generator1[ii](x) + x_tmp)
        
        y = self.generator[-2](x)
        x = self.act(y)
        # x = torch.sin(y)

        y = self.generator[-1](x)
        x = self.actout(y)/math.e
        #print(x.shape)
        #output = output.squeeze(2)
        
        return x, coords

    # def forward(self, coords):
    #     coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input

    #     output, coords = self.out(coords)
    #     return output, coords


class Model():
    def __init__(self, ModelPath, dim, scale_factor, mode, renderer=None, device='cpu'):

        # ======================= JSON Template =======================
        self.Params = {}
        self.Params['ModelPath'] = ModelPath
        # self.Params['DataPath'] = DataPath
        self.renderer = renderer
        self.dim = dim
        self.scale_factor = scale_factor
        current_time = datetime.utcnow()-timedelta(hours=4)
        self.folder = self.Params['ModelPath']+"/"+current_time.strftime("%m_%d_%H_%M")
        # self.folder = None
        # ===== Find next available RUN_X folder =====
        base_path = ModelPath
        run_id = 0

        while True:
            run_folder = os.path.join(base_path, f"Chance_constrained_run_{run_id}")
            if not os.path.exists(run_folder):
                os.makedirs(run_folder)
                break
            run_id += 1

        self.folder = run_folder
        self.folder = None
        # Pass the JSON information
        self.Params['Device'] = device
        self.Params['Pytorch Amp (bool)'] = False

        self.Params['Network'] = {}
        self.Params['Network']['Normlisation'] = 'OffsetMinMax'

        self.Params['Training'] = {}
        self.Params['Training']['Number of sample points'] = 2e5
        self.Params['Training']['Batch Size'] = 2000
        self.Params['Training']['Validation Percentage'] = 10
        self.Params['Training']['Number of Epochs'] = 20000
        self.Params['Training']['Resampling Bounds'] = [0.1, 0.9]
        self.Params['Training']['Print Every * Epoch'] = 1
        self.Params['Training']['Save Every * Epoch'] = 50
        self.Params['Training']['Learning Rate'] = 1e-3#5e-5
        self.Params['Training']['Random Distance Sampling'] = True
        self.Params['Training']['Use Scheduler (bool)'] = False

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss = []
        self.epoch = 0
        self.frame_idx = 0
        self.trajectory = None
        self.all_optimized_trajectories = None
        self.all_optimized_segments = None
        self.prev_state_queue = []
        self.prev_optimizer_queue = []
        self.timer = []
        self.prev_positions = []
        # # Parameters for the occupancy grid
        dim_cells = 100
        limit = 1.05
        spacing = limit/dim_cells
        self.occ_map = gridmap.OccupancyGridMap(np.zeros((dim_cells,dim_cells)), cell_size=spacing, occupancy_threshold=0.8, offset=limit/2)

        self.mode = mode
        self.frame_buffer_size = 20
        self.camera_steps = 5000//50
        self.minimum = 0.007 #0.02
        self.maximum = 0.0156  #0.1
        self.all_framedata = None
        self.all_surf_pc = []
        self.free_pc = []

        self.init_network()
        self.dist_model, self.dist_device = load_sdf_2d_model()

        # keep everything on the same device as the PINN
        self.dist_model = self.dist_model.to(self.Params['Device'])
        self.dist_device = self.Params['Device']
        self.currently_traversed = None
        # self.fixed_start = torch.tensor([-0.3, -0.2, 0.0], dtype=torch.float32)
        # self.fixed_start = self.fixed_start.to(self.Params['Device'])
        # ===== Fixed experiment setup =====
        #####################    MAP1      ############################
        self.fixed_goal = torch.tensor([ 0.0516,  -0.076, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([0.03597647, -0.19569747, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([0.12612747, 0.205, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([-0.1786,   0.12596,  0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([0.15,  0.0396, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([0.2816,   0.1116,  0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.05626768, -0.00738018, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([0.31,  0.161, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([0.31,  0.261, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.186, -0.086, 0], dtype=torch.float32)

        ######################    MAP2      ############################

        self.fixed_goal = torch.tensor([0.0186, 0.06, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([0.062, -0.026, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.0296, -0.236, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.055632,  -0.1266, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.009981,  0.339293, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([0.0397859,  -0.000756, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([0.0770818,  -0.0360409, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.0615918, 0.21549, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([0.13926, 0.2283, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([0.089705, 0.365, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([0.06297, 0.2813, 0.0], dtype=torch.float32)

        ######################    Allensville      ############################
        self.fixed_goal = torch.tensor([-0.1892, -0.0852, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([-0.2712, 0.098632, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([0.01468, -0.20258, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.2762, -0.2695, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([0.1437, -0.0755, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.22977, -0.085229, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.15836, -0.125869, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.229774, -0.086229, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.27321, -0.0265, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.136399, -0.1041596, 0.0], dtype=torch.float32)

        ######################    Denmark      ############################
        # self.fixed_goal = torch.tensor([-0.1509, 0.103031, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.2605, 0.03898, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.130732, 0.0708056, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.036236, 0.215998, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.263, 0.0489, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.266119, 0.02896, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.04500, -0.0395, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.31192, -0.0272235, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.039965, 0.028565, 0.0], dtype=torch.float32)
        # self.fixed_goal = torch.tensor([-0.296277, 0.02713, 0.0], dtype=torch.float32)
        ######################    Superior      ############################
        self.fixed_goal = torch.tensor([-0.129882, -0.147299, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([-0.1362, 0.01396, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([-0.066432, -0.057865, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([0.0322562, -0.148725, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([0.013853, -0.254623, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([0.069989, 0.0141856, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([0.11836, 0.05037, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([0.050712, 0.21064, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([0.22915, 0.0680198, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([0.1217, -0.0936, 0.0], dtype=torch.float32)
        self.fixed_goal = torch.tensor([0.19131, 0.00039, 0.0], dtype=torch.float32)
        self.fixed_goal  = self.fixed_goal.to(self.Params['Device'])
        self.enable_plot = False
    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y)                                                                 

        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x                                                                                                    
    
    def Loss(self, points, Yobs, beta, gamma):
        
      
        tau, Xp = self.network.out(points)
        dtau = self.gradient(tau, Xp)

        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)#torch.norm(D, p=2, dim =1)**2
        
        
        DT0 = dtau[:,:self.dim]
        DT1 = dtau[:,self.dim:]

        T3    = tau[:,0]**2

        LogTau = torch.log(tau[:,0])
        LogTau2 = LogTau**2
        LogTau3 = LogTau**3
        LogTau4 = LogTau**4

        #print(tau.shape)

        T01    = 4*LogTau2*T0/T3*torch.einsum('ij,ij->i', DT0, DT0)
        T02    = -4*LogTau3/tau[:,0]*torch.einsum('ij,ij->i', DT0, D)
        #print(T02.shape)
        T11    = 4*LogTau2*T0/T3*torch.einsum('ij,ij->i', DT1, DT1)
        T12    = 4*LogTau3/tau[:,0]*torch.einsum('ij,ij->i', DT1, D)

        lap0 = 0.5*torch.einsum('ij,ij->i', DT0, DT0)#ltau[:,:self.dim].sum(-1) 
        lap1 = 0.5*torch.einsum('ij,ij->i', DT1, DT1)#ltau[:,self.dim:].sum(-1) 
        
        
        
        S0 = (T01+T02+LogTau4)
        S1 = (T11+T12+LogTau4)
       
        #0.001
        Ypred0 = torch.sqrt(S0)#torch.sqrt
        Ypred1 = torch.sqrt(S1)#torch.sqrt


        Ypred0_visco = Ypred0#
        Ypred1_visco = Ypred1#+gamma*lap1

        sq_Ypred0 = (Ypred0_visco)#+gamma*lap0
        sq_Ypred1 = (Ypred1_visco)#+gamma*lap1


        sq_Yobs0 = (Yobs[:,0])#**2
        sq_Yobs1 = (Yobs[:,1])#**2

        # loss0 = (sq_Yobs0/sq_Ypred0+sq_Ypred0/sq_Yobs0)#**2#+gamma*lap0
        # loss1 = (sq_Yobs1/sq_Ypred1+sq_Ypred1/sq_Yobs1)#**2#+gamma*lap1

        # diff = loss0 + loss1-4
        # loss_n = torch.sum((loss0 + loss1-4))/Yobs.shape[0]
        l0 = sq_Yobs0*(sq_Ypred0)
        l1 = sq_Yobs1*(sq_Ypred1)
        # l0 = torch.sqrt(torch.sqrt(l0))
        # l1 = torch.sqrt(torch.sqrt(l1))

        l0_2 = torch.sqrt(l0)
        l1_2 = torch.sqrt(l1)
        loss0 = (l0_2-1)**2#+gamma*lap0#**2
        loss1 = (l1_2-1)**2

        # loss0 = abs(l0-1)#+gamma*lap0#**2
        # loss1 = abs(l1-1)

        # loss0 = (l0+1/l0-2)#+gamma*lap0#**2
        # loss1 = (l1+1/l1-2)
        diff = loss0 + loss1
        loss_n = torch.sum((loss0 + loss1))/Yobs.shape[0]

        
        loss = beta*loss_n #+ 1e-4*(reg_tau)

        return loss, loss_n, diff

    def init_network(self):
        #! Initialising the network
        #self._init_network()
        self.B = 0.5 * torch.normal(0,1,size=(128,self.dim)) #0.5
        #torch.save(B, self.Params['ModelPath']+'/B.pt')

        self.network = NN(self.Params['Device'],self.dim, self.B)
        self.network.apply(self.network.init_weights)
        #self.network.float()
        self.network.to(self.Params['Device'])

        #! Defining the optimization scheme
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.Params['Training']['Learning Rate']
            ,weight_decay=0.1)
        if self.Params['Training']['Use Scheduler (bool)'] == True:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
            #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000,2000], gamma=0.5)

    def load_rawdata(self):
        #! load data
        ################## Map 1 ##################
        initial_view = Tensor([-0.12, -0.046, 0])
        initial_view = Tensor([ 0.042176,  -0.02060163, 0.0])
        # initial_view = Tensor([0.3608916,  0.171, 0.0]) #initial_view = Tensor([0.36,  0.151, 0.0])
        initial_view = Tensor([-0.05826768, -0.00738018, 0.0])
        # initial_view = Tensor([0.07269316, -0.05, 0.0])
        # initial_view = Tensor([0.16602385, 0.0396, 0.0 ])
        # initial_view = Tensor([-0.1886,   0.12296,  0.0])
        # initial_view = Tensor([0.19120717,   0.1996912,  0.0])
        # initial_view = Tensor([0.1912,   0.1996723,  0.0])
        # initial_view = Tensor([ 0.0096,  -0.0186, 0.0])
        ################## Map 2 ##################
        initial_view = Tensor([ 0.0616,  -0.0216, 0.0])
        initial_view = Tensor([ 0.01103258,  -0.116, 0.0])
        # initial_view = Tensor([ -0.0036,  -0.0326, 0.0])
        # initial_view = Tensor([0.02538, -0.22976, 0.0])
        # initial_view = Tensor([0.15026, 0.23365, 0.0])
        # initial_view = Tensor([-0.0136337, 0.145519, 0.0])
        # initial_view = Tensor([-0.0491679, -0.129549, 0.0])
        # initial_view = Tensor([0.06298, 0.281353, 0.0])
        # initial_view = Tensor([-0.02029, 0.352, 0.0])
        # initial_view = Tensor([-0.016616, 0.2232728, 0.0])
        # initial_view = Tensor([0.164693, 0.203633, 0.0])
        ################## Allensville ##################
        initial_view = Tensor([0.0536, -0.1866, 0.0])
        initial_view = Tensor([-0.15136, -0.15586, 0.0])
        initial_view = Tensor([-0.22977, -0.085229, 0.0])
        # initial_view = Tensor([-0.3755, -0.3741, 0.0])
        # initial_view = Tensor([-0.00725, -0.31602, 0.0])
        # initial_view = Tensor([-0.29873, -0.15181, 0.0])
        # initial_view = Tensor([-0.306212, 0.160368, 0.0])
        # initial_view = Tensor([-0.25277, 0.14011, 0.0])
        # initial_view = Tensor([-0.16236, -0.123969, 0.0])
        # initial_view = Tensor([-0.30698, -0.29587, 0.0])

        ################## Denmark ##################
        # initial_view = Tensor([-0.03756, -0.01000, 0.0])
        # initial_view = Tensor([-0.0164155, 0.201127, 0.0])
        # initial_view = Tensor([-0.0101138, -0.02388, 0.0])
        # initial_view = Tensor([-0.26375, 0.0504, 0.0])
        # initial_view = Tensor([-0.030326, 0.18998, 0.0])
        # initial_view = Tensor([-0.20589, 0.104213, 0.0])
        # initial_view = Tensor([-0.1027, 0.103383, 0.0])
        # initial_view = Tensor([-0.3620, 0.160228, 0.0])
        # initial_view = Tensor([-0.266721, 0.216986, 0.0])
        # initial_view = Tensor([-0.390377, 0.101086, 0.0])

        ################## Superior ##################
        initial_view = Tensor([-0.0797515, -0.034389, 0.0])
        initial_view = Tensor([-0.08850, 0.0915668, 0.0])
        initial_view = Tensor([-0.0615467, -0.173793, 0.0])
        initial_view = Tensor([0.01909, -0.10586, 0.0])
        initial_view = Tensor([0.02132, -0.102299, 0.0])
        initial_view = Tensor([0.13973, -0.105728, 0.0])
        initial_view = Tensor([0.0344291, 0.0116167, 0.0])
        initial_view = Tensor([0.02166, 0.11691, 0.0])
        initial_view = Tensor([0.176635, 0.09965, 0.0])
        initial_view = Tensor([0.22952, -0.0374228, 0.0])
        initial_view = Tensor([0.236685, -0.040921, 0.0])
        self.initial_view = initial_view
        
        if self.mode == READ_FROM_COOKED_DATA: # read from file
            # all_points = np.load("sampled_points.npy")
            # all_speeds = np.load("speed.npy")
            explored_data = np.load("data/explored_data.npy")
            self.explored_data = torch.tensor(explored_data).to(self.Params['Device'])

        self.cur_view = initial_view

    def randomize_data(self, frame_data):
        prev_frame_idx = torch.randperm(self.frame_idx).tolist()[:self.frame_buffer_size-1]
        prev_framedata = self.all_framedata[prev_frame_idx]
        chosen_framedata = torch.cat((prev_framedata, frame_data.unsqueeze(0)), dim=0).view(-1, 8)

        chosen_points = chosen_framedata[:,:6]
        chosen_speeds = chosen_framedata[:,6:]
        
        points = frame_data[:,:6]
        speeds = frame_data[:,6:]
        if False: # ray sampling method
            local_start_index = torch.randperm(points.shape[0])[:int(points.shape[0]*1.0)]
            local_end_index = torch.randperm(points.shape[0])[:int(points.shape[0]*1.0)]
            local_points_combination = torch.cat((points[local_start_index,:3], points[local_end_index,3:]), dim=1)
            local_speeds_combination = torch.cat((speeds[local_start_index,:1], speeds[local_end_index,1:]), dim=1)
        else: # random sampling method
            local_points_combination = torch.cat((points[:,:3], points[:,3:]), dim=1)
            local_speeds_combination = torch.cat((speeds[:,:1], speeds[:,1:]), dim=1)

        chosen_percent = 1.0
        # print("chosen points shape:", chosen_points.shape)
        chosen_start_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
        chosen_end_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
        # print("chosen points indice shape:", chosen_start_index.shape)
        chosen_start_points = chosen_points[chosen_start_index][:, :3]
        chosen_start_speeds = chosen_speeds[chosen_start_index][:, 0]
        chosen_end_points = chosen_points[chosen_end_index][:, 3:6]
        chosen_end_speeds = chosen_speeds[chosen_end_index][:, 1]
        # print("chosen start points shape:", chosen_start_points.shape)
        global_points_combination = torch.cat((chosen_start_points, chosen_end_points), dim=1)
        global_speeds_combination = torch.cat((chosen_start_speeds[:, None], chosen_end_speeds[:, None]), dim=1)

        all_points = torch.cat((local_points_combination, global_points_combination), dim=0)
        all_speeds = torch.cat((local_speeds_combination, global_speeds_combination), dim=0)

        cur_data = torch.cat((all_points, all_speeds), dim=1)
        return cur_data
    def is_stuck(self, threshold=0.005):
        if len(self.prev_positions) < 2:
            return False

        start = self.prev_positions[-2]
        end   = self.prev_positions[-1]

        movement = np.linalg.norm(end[:2] - start[:2])

        return movement < threshold
    def train(self):
        if self.folder is not None and not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.load_rawdata()
        is_one_frame = True
        policy_times = []           # <-- add
        optimizer_times = []        # <-- add
        total_planning_times = []
        while True:
            if True:
                print("Current Viewpoint:", self.cur_view)

                #! generate data start
                if self.mode == EXPLORATION:
                    heights = [0]
                    curpoints = [] 
                    curspeeds = []
                    curbounds = []
                    for height in heights:
                        curviewpoint = self.cur_view.clone()
                        curviewpoint[2] = height
                        points, speeds, bounds, obstacle_points = igib_runner.sample_points_and_speeds_from_pos_neural(self, curviewpoint.cpu().numpy(), self.minimum, self.maximum, num=10000, scale_factor=self.scale_factor)
                        curpoints.append(points)
                        curspeeds.append(speeds)
                        curbounds.append(bounds)          
                    frame_points = torch.cat(curpoints, dim=0)
                    frame_speeds = torch.cat(curspeeds, dim=0)
                    frame_bounds = torch.cat(curbounds, dim=0)
                    frame_epoch = 50

                    #? save occupancy map
                    # self.occ_map.save(self.folder+"/occ_map.npz")
                elif self.mode == READ_FROM_COOKED_DATA:
                    if self.frame_idx >= self.explored_data.shape[0]:
                        break
                    heights = [0]
                    curpoints = [] 
                    curspeeds = []
                    curbounds = []         
                    frame_points = self.explored_data[self.frame_idx, :, :6]
                    frame_speeds = self.explored_data[self.frame_idx, :, 6:]
                    frame_epoch = 50



                #! generate data end
                            
                frame_data = torch.cat((frame_points.to(self.Params['Device']), frame_speeds.to(self.Params['Device'])), dim=1)

            ####################! Core Training #############################
            self.alpha = 1.025
            total_diff, cur_data = self.train_core(frame_epoch, frame_data, is_one_frame)



            #! Policy
            if self.mode != READ_FROM_COOKED_DATA:
                # nbv = self.policy_greedy(pre_view, cur_data.detach(), gamma)
                # nbv = self.policy_handcrafted(frame_idx)
                traj_list = None
                if self.mode == EXPLORATION:
                    #! Update the occupancy grid
                    #? filter frame points with some z range
                    valid = (frame_points[:,-1] > -0.1) & (frame_points[:,-1] < 0.1)
                    self.occ_map.update(self.cur_view.clone().cpu().numpy(), frame_points[valid].cpu().numpy(), frame_bounds[valid].cpu().numpy())
                    coverage = self.occ_map.get_coverage()
                    print("Occupancy Grid Coverage:", coverage)
                    # if coverage > 0.543:
                    #     # pass
                    #     break
                    if self.reached_goal(self.cur_view) or self.is_stuck():
                        print("Reached goal! Stopping exploration.")
                        length = self.compute_path_length(self.trajectory)
                        print(f"Total traversed length: {length * self.scale_factor:.4f} m")
                        # ===== SAVE FULL TRAJECTORY =====
                        if self.trajectory is not None:
                            if self.folder is not None:
                                base_name = "full_trajectory"
                                i = 0
                                while True:
                                    save_path = os.path.join(self.folder, f"{base_name}_{i}.npy")
                                    if not os.path.exists(save_path):
                                        break
                                    i += 1
                                np.save(save_path, self.trajectory)
                                print(f"Saved full trajectory to: {save_path}")
                            if self.enable_plot:
                                self.plot(
                                    self.initial_view,                 # start
                                    self.fixed_goal,                   # goal
                                    self.epoch,
                                    total_diff.item(),
                                    self.alpha,
                                    cur_data[:,:6].clone().cpu().numpy(),
                                    None,
                                    None,
                                    obstacle_points,
                                    None, #np.array(self.all_optimized_trajectories) if self.all_optimized_trajectories is not None else None,
                                    final=True
                                )
                            return {
                                "length": length,
                                "collision": False,
                                "policy_times": policy_times,
                                "optimizer_times": optimizer_times,
                                "total_planning_times": total_planning_times,
                            }
                        # break
                    # traj_list, traj_ind = self.policy_occ(self.cur_view.detach().clone().cpu().numpy(), height=0) 
                    # --- policy timing ---
                    t0 = time.perf_counter()
                    traj_list, traj_ind = self.policy_goal_direct(self.cur_view.detach().clone().cpu().numpy(), height=0) 
                    t1 = time.perf_counter()
                    # traj_list is the full gradient-descent trajectory from current position to the selected unexplored block. 
                    # traj_ind is the index along the trajectory where accumulated path length ≈ 0.05 meters.
                    nbv = Tensor(traj_list[traj_ind])


                # print("*"*10)
                # print("self.dataset.Ts[0][:3, 3]/self.scale_factor", self.dataset.Ts[0][:3, 3]/self.scale_factor)
                # print("nbv", nbv)
                # print("curview", self.cur_view)
                # traj = self.predict_trajectory2(self.cur_view.detach().clone().cpu().numpy(), nbv.detach().clone().cpu().numpy()) # small incremental motion
               

                # Optimize the full trajectory
                start = traj_list[0]
                path = traj_list[1:]
                profile_this = (self.frame_idx == 0)

                if profile_this:
                    # Create the profile directory once per run, reuse for any subsequent dumps
                    if not hasattr(self, "_profile_dir"):
                        stamp = (datetime.utcnow() - timedelta(hours=4)).strftime("%m_%d_%H_%M")
                        self._profile_dir = f"/antfields/time_profiles/time_profile_{stamp}"
                        os.makedirs(self._profile_dir, exist_ok=True)
                        print(f"[profile] saving profiles to {self._profile_dir}")

                    profiler = cProfile.Profile()
                    profiler.enable()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                t2 = time.perf_counter()
                optimized_traj_list = rollout_optimized(
                    start,
                    path,
                    obstacle_points,
                    solve_step,
                    self.dist_model,
                    self.Params['Device'],
                    epoch = self.epoch,
                    folder=self.folder
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
                if profile_this:
                    profiler.disable()
                    prof_path = f"{self._profile_dir}/rollout_profile_epoch_{self.epoch}.prof"
                    profiler.dump_stats(prof_path)
                    print(f"[profile] saved {prof_path}")

                    # Also dump a readable text summary alongside the .prof
                    txt_path = f"{self._profile_dir}/rollout_profile_epoch_{self.epoch}.txt"
                    with open(txt_path, "w") as f:
                        ps = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
                        ps.print_stats(40)
                        f.write("\n--- by tottime (self-time, excludes callees) ---\n")
                        ps = pstats.Stats(profiler, stream=f).sort_stats("tottime")
                        ps.print_stats(30)

                    # And print a short version inline for quick eyeballing
                    s = io.StringIO()
                    pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(20)
                    print(s.getvalue())


                # --- compute durations ---
                policy_time = t1 - t0
                optimizer_time = t3 - t2
                total_time = (t1 - t0) + (t3 - t2)

                print(f"Policy time: {policy_time:.4f}s")
                print(f"Optimizer time: {optimizer_time:.4f}s")
                print(f"Total planning time: {total_time:.4f}s")
                policy_times.append(policy_time)             # <-- add
                optimizer_times.append(optimizer_time)       # <-- add
                total_planning_times.append(total_time)      # <-- add
                optimized_traj_list = [
                    p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)
                    for p in optimized_traj_list
                ]
                
                traj_list = np.array(traj_list)
                print("traj_list size is: ", traj_list.shape)
                print("optimized_traj_list size is: ", len(optimized_traj_list))
                if self.mode == EXPLORATION:
                    optimized_segment = optimized_traj_list[:traj_ind+1]
                # Now NBV is taken from the optimized trajectory
                # nbv = Tensor(traj_list[traj_ind])
                nbv = Tensor(optimized_traj_list[traj_ind])
                if self.all_optimized_trajectories is None:
                    self.all_optimized_trajectories = optimized_traj_list
                if self.all_optimized_segments is None:
                    self.all_optimized_segments = optimized_segment
                else:
                    # self.all_optimized_segments = np.concatenate([self.all_optimized_segments, np.array(optimized_segment)],axis=0)
                    optimized_segment_np = np.array([
                        p.detach().cpu().numpy() if torch.is_tensor(p) else p
                        for p in optimized_segment
                    ])
                    self.currently_traversed = self.all_optimized_segments.copy()
                    self.all_optimized_segments = np.concatenate(
                        [self.all_optimized_segments, optimized_segment_np],
                        axis=0
                    )
                if self.mode == EXPLORATION and self.folder is not None:
                    traj = traj_list[:traj_ind+1]
                    plot_traj_difference(
                        self.folder,
                        self.epoch,
                        traj,
                        optimized_segment,
                        self
                    )
                # if self.trajectory is None:
                #     self.trajectory = traj #stores the entire path history over time.
                # else:
                #     self.trajectory = np.concatenate([self.trajectory, traj], axis=0) 
                segment_np = np.array([
                    p.detach().cpu().numpy() if torch.is_tensor(p) else p
                    for p in optimized_segment
                ])

                if self.trajectory is None:
                    self.trajectory = segment_np
                else:
                    self.trajectory = np.concatenate([self.trajectory, segment_np[1:]], axis=0)
                # print("traj is:", traj)
                collision, idxs = check_collision_with_surface_points(
                    segment_np,
                    obstacle_points,
                    robot_radius=0.0105,
                    safety_margin=0.00001,
                    return_details=True
                )

                if collision:
                    print(f"⚠️ Collision detected at indices: {idxs}")

                    save_path = os.path.join(self.folder, f"collision_traj_step_{self.frame_idx}.npy")
                    np.save(save_path, self.trajectory)

                    return {
                        "length": None,
                        "collision": True,
                        "policy_times": policy_times,
                        "optimizer_times": optimizer_times,
                        "total_planning_times": total_planning_times,
                    }
                
                #? ******************SAVINGS start*******************
                save_traj = True
                if save_traj and self.folder is not None:
                    np.save(self.folder+"/traj"+"_"+str(self.epoch)+".npy", self.trajectory)


                
                camera_matrix = None
                if self.enable_plot:
                    self.plot(self.initial_view, nbv, self.epoch, total_diff.item(),self.alpha, cur_data[:,:6].clone().cpu().numpy(), camera_matrix, traj_list, obstacle_points, np.array(optimized_traj_list))
                
            elif self.mode == READ_FROM_COOKED_DATA:
                self.plot(self.initial_view, np.array([0.3, 0.2, 0]), self.epoch, total_diff.item(),self.alpha, cur_data[:,:6].clone().cpu().numpy(), None)

            
            # self.fourplot(epoch, FRAMES[:frame_idx+1], total_diff.item(), alpha)
            with torch.no_grad():
                if self.folder is not None:
                    self.save(epoch=self.epoch, val_loss=total_diff)

            #? ******************SAVINGS end*******************
            
            self.frame_idx += 1
            if self.frame_idx >= self.camera_steps:
                break
            
            if self.mode != READ_FROM_COOKED_DATA:
                # self.cur_view = nbv
                self.cur_view = torch.as_tensor(
                    optimized_segment[-1],
                    dtype=torch.float32,
                    device=self.Params['Device']
                )
                self.prev_positions.append(self.cur_view.detach().cpu().numpy())

                if len(self.prev_positions) > 10:
                    self.prev_positions.pop(0)
        if False:
            print("Exploration is done. Finetuning...")
            self.plot(self.cur_view, self.cur_view, self.epoch, total_diff.item(),self.alpha)
            #?: train for another 1000 epochs for finetuning
            # 1. data is all framedata
            prev_frame_idx = torch.randperm(self.frame_idx).tolist()[:self.frame_buffer_size-1]
            #! mix data so that the start and end points are from different frames
            prev_framedata = self.all_framedata[prev_frame_idx]
            chosen_percent = 1.0
            # print("chosen points shape:", chosen_points.shape)
            chosen_framedata = torch.cat((prev_framedata, frame_data.unsqueeze(0)), dim=0).view(-1, 8)

            chosen_points = chosen_framedata[:,:6]
            chosen_speeds = chosen_framedata[:,6:]
            chosen_start_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
            chosen_end_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
            # print("chosen points indice shape:", chosen_start_index.shape)
            chosen_start_points = chosen_points[chosen_start_index][:, :3]
            chosen_start_speeds = chosen_speeds[chosen_start_index][:, 0]
            chosen_end_points = chosen_points[chosen_end_index][:, 3:6]
            chosen_end_speeds = chosen_speeds[chosen_end_index][:, 1]
            # print("chosen start points shape:", chosen_start_points.shape)
            global_points_combination = torch.cat((chosen_start_points, chosen_end_points), dim=1)
            global_speeds_combination = torch.cat((chosen_start_speeds[:, None], chosen_end_speeds[:, None]), dim=1)
            fine_data = torch.cat((global_points_combination, global_speeds_combination), dim=1)

            # 2. core training
            fine_epoch = 1000
            total_diff, cur_data = self.train_core(fine_epoch, fine_data, False)

            # 3. save a final plot #
            self.plot(self.cur_view, self.cur_view, self.epoch, total_diff.item(),self.alpha)
            with torch.no_grad():
                self.save(epoch=self.epoch, val_loss=total_diff)

        return None
    def train_core(self, epoch, frame_data, is_one_frame=True):
        beta = 1.0
        prev_diff = 1.0
        current_diff = 1.0
        step = -2000.0/4000.0
        step = 0
        tt =time.time()

        if is_one_frame:
            if self.all_framedata is None:
                self.all_framedata = frame_data.unsqueeze(0)
            else:
                self.all_framedata = torch.cat((self.all_framedata, frame_data.unsqueeze(0)), dim=0)
            print(self.all_framedata.shape)
            if self.folder is not None:
                np.save(f"{self.folder}/explored_data.npy", self.all_framedata.clone().cpu().numpy())


        #! mix data so that the start and end points are from different frames
        if is_one_frame:
            cur_data = self.randomize_data(frame_data)
            # print("cur_data:", cur_data.shape)
        else:
            rand_idx = torch.randperm(frame_data.shape[0])
            chosen_data = frame_data[rand_idx][:100000]
            chosen_points = chosen_data[:,:6]
            chosen_speeds = chosen_data[:,6:]
            
            chosen_percent = 1.0
            # print("chosen points shape:", chosen_points.shape)
            chosen_start_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
            chosen_end_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
            # print("chosen points indice shape:", chosen_start_index.shape)
            chosen_start_points = chosen_points[chosen_start_index][:, :3]
            chosen_start_speeds = chosen_speeds[chosen_start_index][:, 0]
            chosen_end_points = chosen_points[chosen_end_index][:, 3:6]
            chosen_end_speeds = chosen_speeds[chosen_end_index][:, 1]

            cur_data = torch.cat((chosen_start_points, chosen_end_points, chosen_start_speeds[:, None], chosen_end_speeds[:, None]), dim=1)
                
        # if self.mode == READ_FROM_COOKED_DATA: #? use all framedata
        dataloader = FastTensorDataLoader(cur_data, batch_size=int(self.Params['Training']['Batch Size']), shuffle=True)


        # frame_epoch = 10000 if isOffline else 50
        frame_epoch = epoch
        start_time = time.time()
        for e in range(frame_epoch):
            self.epoch += 1
            total_train_loss = 0
            total_diff=0

            # alpha = min(max(0.85,0.85+0.5*step),0.9)
            # alpha = 1.10#1.05
            

            step+=1.0/4000/((int)(self.epoch/4000)+1.) * 2.0
            gamma=0.001#max((4000.0-epoch)/4000.0/20,0.001)

            current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
            current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
            self.prev_state_queue.append(current_state)
            self.prev_optimizer_queue.append(current_optimizer)
            if(len(self.prev_state_queue)>5):
                self.prev_state_queue.pop(0)
                self.prev_optimizer_queue.pop(0)
            
            #self.optimizer.param_groups[0]['lr']  = np.clip(1e-3*(1-(epoch-8000)/1000.), a_min=5e-4, a_max=1e-3) 
            self.optimizer.param_groups[0]['lr']  = 5e-4


            prev_diff = current_diff
            iter=0
            while True:
                total_train_loss = 0
                total_diff = 0
                #for i in range(10):
                for i, data in enumerate(dataloader, 0):#train_loader_wei,dataloader
                    #print('----------------- Epoch {} - Batch {} --------------------'.format(epoch,i))
                    if i>5:
                        break
                    t0 = time.time()
    
                    data=data[0].to(self.Params['Device'])
                    #data, indexbatch = data
                    points = data[:,:2*self.dim]#.float()#.cuda()
                    speed = data[:,2*self.dim:]#.float()#.cuda()
                    
                    speed = speed*speed*(2-speed)*(2-speed)
                    speed = self.alpha*speed+1-self.alpha


                    loss_value, loss_n, wv = self.Loss(
                    points, speed, beta, gamma)
                    loss_value.backward()

                    # Update parameters
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    #print('')
                    #print(loss_value.shape)
                    total_train_loss += loss_value.clone().detach()
                    total_diff += loss_n.clone().detach()
                    t1 = time.time()
                    #print(t1-t0)
                    #print('')
                    #weights[indexbatch] = wv
                    
                    del points, speed, loss_value, loss_n, wv#,indexbatch
                

                total_train_loss /= 5 #dataloader train_loader
                total_diff /= 5 #dataloader train_loader

                current_diff = total_diff
                diff_ratio = current_diff/prev_diff
        
                if True and (diff_ratio < 1.2 and diff_ratio > 0 or e < 10):#1.5
                    #self.optimizer.param_groups[0]['lr'] = prev_lr 
                    break
                else:
                    
                    iter+=1
                    with torch.no_grad():
                        random_number = random.randint(0, 4)
                        self.network.load_state_dict(self.prev_state_queue[random_number], strict=True)
                        self.optimizer.load_state_dict(self.prev_optimizer_queue[random_number])

                    print("RepeatEpoch = {} -- Loss = {:.4e} -- Alpha = {:.4e}".format(
                        self.epoch, total_diff, self.alpha))
                
                
            #'''
            self.total_train_loss.append(total_train_loss)
            
            beta = 1.0/total_diff
            
            t_1=time.time()
            #print(t_1-t_0)

            #del train_loader_wei, train_sampler_wei

            if self.Params['Training']['Use Scheduler (bool)'] == True:
                self.scheduler.step(total_train_loss)

            t_tmp = tt
            tt=time.time()
            #print(tt-t_tmp)
            #print('')
            if self.epoch % self.Params['Training']['Print Every * Epoch'] == 0:
                with torch.no_grad():
                    #print("Epoch = {} -- Training loss = {:.4e} -- Validation loss = {:.4e}".format(
                    #    epoch, total_train_loss, total_val_loss))
                    print("Epoch = {} -- Loss = {:.4e} -- Alpha = {:.4e}".format(
                        self.epoch, total_diff.item(), self.alpha))
        end_time = time.time()
        # if is_one_frame:
        #     duration = end_time-start_time
        #     print(f"running in {duration} secs")
        #     self.timer.append(duration)
        #     np.save("gib6_time.npy",np.array(self.timer))
        return total_diff, cur_data

    def save(self, epoch='', val_loss=''):
        '''
            Saving a instance of the model
        '''
        torch.save({'epoch': epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'B_state_dict':self.B,
                    'train_loss': self.total_train_loss,
                    'val_loss': self.total_val_loss}, '{}/Model_Epoch_{}_ValLoss_{:.6e}.pt'.format(self.folder, str(epoch).zfill(5), val_loss))

    def load(self, filepath):
        #B = torch.load(self.Params['ModelPath']+'/B.pt')
        
        checkpoint = torch.load(
            filepath, map_location=torch.device(self.Params['Device']))
        self.B = checkpoint['B_state_dict']

        self.network = NN(self.Params['Device'],self.dim,self.B)

        self.network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.network.to(torch.device(self.Params['Device']))
        self.network.float()
        self.network.eval()

    def load_traj(self, filepath):
        self.trajectory = np.load(filepath)
     
    def Gradient(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
       
        #Xp.requires_grad_()
        
        #tau, dtau, coords = self.network.out_grad(Xp)

        tau, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D))

        LogTau = torch.log(tau[:,0])
        LogTau2 = LogTau**2

        A = 2*LogTau*T0/tau[:,0]
        B = LogTau2/T0
        #print(A.shape)

        Ypred0 = -A*dtau[:,:self.dim]+B*D
        #print(Ypred0.shape)
        Spred0 = torch.norm(Ypred0)
        Ypred0 = 1/Spred0**2*Ypred0

        Ypred1 = -A*dtau[:,self.dim:]-B*D
        Spred1 = torch.norm(Ypred1)

        Ypred1 = 1/Spred1**2*Ypred1
        
        return torch.cat((Ypred0, Ypred1),dim=1)

    def TravelTimes(self, Xp):
        # Apply projection from LatLong to UTM
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        tau, coords = self.network.out(Xp)

       
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = torch.log(tau[:, 0])**2* torch.sqrt(T0)

        del Xp, tau, T0
        return TT
    
    def Tau(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
     
        tau, coords = self.network.out(Xp)

        return tau

    def Speed(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))

        tau, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        #Xp.requires_grad_()
        #tau, dtau, coords = self.network.out_grad(Xp)

        
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        T0 = torch.einsum('ij,ij->i', D, D)

        DT1 = dtau[:,self.dim:]

        T3    = tau[:,0]**2

        LogTau = torch.log(tau[:,0])
        LogTau2 = LogTau**2
        LogTau3 = LogTau**3
        LogTau4 = LogTau**4

        T1    = 4*LogTau2*T0/T3*torch.einsum('ij,ij->i', DT1, DT1)
        T2    = 4*LogTau3/tau[:,0]*torch.einsum('ij,ij->i', DT1, D)

        
        
        S = (T1+T2+LogTau4)

        Ypred = 1/torch.sqrt(S)
        
        del Xp, tau, dtau, T0, T1, T2, T3, LogTau,LogTau2,LogTau3,LogTau4,DT1,S
        return Ypred


    def plot(self, src, tar, epoch, total_train_loss, alpha, cur_points=None, camera_matrix=None, traj_list = None, obstacle_points=None, optimized_path=None, final=False):
        # obstacle_points = None
        limit = 1
        xmin = [-0.5, -0.5]
        xmax = [0.5, 0.5]
        spacing=limit/180.0
        X,Y      = np.meshgrid(np.arange(xmin[0],xmax[0],spacing),np.arange(xmin[1],xmax[1],spacing))

        Xsrc = src 
        # Xsrc[2] = 0.2
        # Xsrc = [0.2, -0.2, 0]
        
        XP       = np.zeros((len(X.flatten()),2*self.dim))#*((xmax[dims_n]-xmin[dims_n])/2 +xmin[dims_n])
        XP[:,:self.dim] = Xsrc
        XP[:,self.dim:] = Xsrc
        #XP=XP/scale
        XP[:,self.dim+0]  = X.flatten()
        XP[:,self.dim+1]  = Y.flatten() #! this allow to change from y to z
        XP = Variable(Tensor(XP)).to(self.Params['Device'])
        
        tt = self.TravelTimes(XP)
        ss = self.Speed(XP)#*5
        tau = self.Tau(XP)
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        V  = ss.to('cpu').data.numpy().reshape(X.shape)
        TAU = tau.to('cpu').data.numpy().reshape(X.shape)
        np.savez(
            self.folder + f"/field_epoch_{epoch}.npz",
            X=X,
            Y=Y,
            speed=V,
            travel_time=TT
        )
        fig = plt.figure()

        ax = fig.add_subplot(111)
        # ax.invert_xaxis()
        # ax.invert_yaxis()
        quad1 = ax.pcolormesh(X,Y,V,vmin=0,vmax=1)


        #! camera triangle
        if camera_matrix is not None:
            orientation_matrix = camera_matrix[:3, :3]  # Assuming the rotation part is the 3x3 upper-left submatrix
            position = src

            # Calculate the orientation angle from the orientation matrix
            yaw = np.arctan2(orientation_matrix[1, 0], orientation_matrix[0, 0])  # You can use this angle to represent the camera look at
            yaw += np.pi/2

            # Create a small triangle marker for the camera
            triangle_size = 0.2  # Adjust the size as needed
            triangle_marker = np.array([[0, 0], [triangle_size/2, -triangle_size/2], [triangle_size / 2, triangle_size/2]])
            
            # Rotate the triangle marker to match the camera's orientation
            rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rotated_triangle_marker = np.dot(triangle_marker, rotation_matrix.T)

            # # Add the triangle marker at the camera position
            # ax.plot(rotated_triangle_marker[0], rotated_triangle_marker[1], 'r^', markersize=10)

            camera_x = position[0] + rotated_triangle_marker[:, 0]
            camera_y = position[1] + rotated_triangle_marker[:, 1]
            ax.fill(camera_x, camera_y, 'b')

        src_np = src if isinstance(src, np.ndarray) else src.cpu().numpy()
        tar_np = tar if isinstance(self.fixed_goal, np.ndarray) else self.fixed_goal.cpu().numpy()
        # ---- Zoom based on start + goal ONLY ----
        pts = np.vstack((src_np[:2], tar_np[:2]))

        padding = 0.1  # adjust if needed
        xmin = np.min(pts[:, 0]) - padding
        xmax = np.max(pts[:, 0]) + padding
        ymin = np.min(pts[:, 1]) - padding + 0.02
        ymax = np.max(pts[:, 1]) + padding

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        # robot_circle = plt.Circle(
        #     (src_np[0], src_np[1]),
        #     0.105/10,
        #     fill=False,
        #     edgecolor='green',
        #     linewidth=2,
        #     label='start point',
        #     zorder=10
        # )
        # ax.add_patch(robot_circle)

        # ax.scatter(src_np[0], src_np[1], color='red', s=20, zorder=11)
        cur_np = self.cur_view.detach().cpu().numpy()
        if final is True:
            cur_np = tar_np
        ax.scatter(
            src_np[0],
            src_np[1],
            color='cyan',        # high contrast vs green background
            s=40,
            marker='o',
            edgecolor='black',
            linewidth=0.5,
            label='start',
            zorder=12
        )
        robot_radius = 0.0105
        circle = plt.Circle(
            (cur_np[0], cur_np[1]),
            robot_radius,
            color='black',
            fill=False,
            linewidth=2,
            zorder=11
        )
        ax.add_patch(circle)
        # ax.scatter(
        #     cur_np[0],
        #     cur_np[1],
        #     color='black',
        #     s=40,
        #     label='robot',
        #     zorder=11
        # )
        ax.scatter(
            tar_np[0],
            tar_np[1],
            color='magenta',
            s=60,
            marker='*',
            label='goal',
            zorder=12
        )
        #! plot trajectory
        if traj_list is not None:
            # ax.plot(traj_list[:, 0], traj_list[:, 1], color='pink', marker = 'o', markersize=0.8, label='unoptimized planned', linestyle='-')
            traj_list_path = self.folder+"/"+str(epoch)+".npy"
            # np.save(traj_list_path, traj_list)
            plt.savefig(self.folder+"/plots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.png",bbox_inches='tight')

            #? plot trajectory with step size
            # ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], color='red', marker='o', markersize=0.8, linestyle='-', linewidth=1)
        if optimized_path is not None:
            # ax.plot(optimized_path[:, 0], optimized_path[:, 1], color=(0.7, 0.9, 1.0), marker = 'o', label='optimized planned', markersize=0.8, linestyle='-')
            ax.plot(optimized_path[:, 0], optimized_path[:, 1], color='pink', marker = 'o', label='optimized planned', markersize=0.8, linestyle='-')
        # plot optimized segments (blue)
        # if self.all_optimized_segments is not None:
        #     seg = self.all_optimized_segments[:-1]
        if self.currently_traversed is not None:
            seg = self.currently_traversed
            if isinstance(seg, torch.Tensor):
                seg = seg.detach().cpu().numpy()
            if isinstance(seg, list):
                seg = np.array([
                    p.detach().cpu().numpy() if torch.is_tensor(p) else p
                    for p in seg
                ])
            ax.plot(seg[:, 0], seg[:, 1], color='blue', marker='o', markersize=0.8, linestyle='-', linewidth=1)  
            if final:
                ax.plot(
                    [seg[-1][0], tar_np[0]],
                    [seg[-1][1], tar_np[1]],
                    color='blue',
                    linewidth=1
                )
        if obstacle_points is not None:
            if torch.is_tensor(obstacle_points):
                obstacle_points = obstacle_points.detach().cpu().numpy()
                np.save(self.folder + f"/obstacle_points_{epoch}.npy", obstacle_points)
            ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], c='red', s=0.5, alpha=0.5, label="surface points", zorder=10)
        
        ax.contour(X,Y,TT,np.arange(0,5,0.02), cmap='bone', linewidths=0.3)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        ax.set_title(f"Online Planning Step {epoch}", fontsize=11)
        if final:
            ax.set_title(f"Reached Goal at Step {epoch}", fontsize=11)
        robot_legend = Line2D(
            [0], [0],
            color='black',
            marker='o',
            linestyle='None',
            markersize=8,
            markerfacecolor='none',   # hollow circle
            markeredgewidth=1.5,
            label='robot'
        )

        handles, labels = ax.get_legend_handles_labels()
        handles.append(robot_legend)
        labels.append('robot')

        # ax.legend(handles, labels, loc='upper right', fontsize=8)
        ax.legend(
            handles, labels,
            loc='lower left',              # anchor point of legend box
            bbox_to_anchor=(1.02, 1.0),    # move it outside to the right
            fontsize=8,
            borderaxespad=0.
        )
        # ax.legend(loc='upper right', fontsize=8)
        plt.savefig(self.folder+"/plots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.png",bbox_inches='tight')

        plt.close(fig)

        # draw occupancy grid
        if self.mode in [EXPLORATION]:
            pathname = self.folder+"/plots"+str(epoch)+"_"+"occ"+".png"
            cur_small_loc = src.cpu().numpy()
            tar_small_loc = tar.cpu().numpy()
            self.occ_map.plot(cur_small_loc, tar_small_loc, path=pathname)
            

        if cur_points is not None:
            # print("plot: cur point shape:",cur_points.shape)
            start_points = cur_points[:,:3]
            end_points = cur_points[:,3:]
            diff = start_points - end_points
            dists = np.linalg.norm(diff, axis=-1)
            # print("dists:", dists.shape)
            plt.hist(dists, bins=100)
            plt.savefig(self.folder+"/plots_dist_"+str(epoch)+".png")
            plt.close()
        if self.trajectory is not None and self.all_optimized_segments is not None:

            traj = np.asarray(self.trajectory)
            opt  = np.asarray(self.all_optimized_segments)

            n = min(len(traj), len(opt))
            traj = traj[:n]
            opt  = opt[:n]

            idx = find_deviation_region(traj, opt)

            if idx is not None:

                pts = np.vstack((traj[idx], opt[idx]))

                xmin, xmax, ymin, ymax = compute_zoom_bounds(pts, padding=0.15)

                fig = plt.figure()
                ax = fig.add_subplot(111)

                quad1 = ax.pcolormesh(X, Y, V, vmin=0, vmax=1)

                # nominal path
                ax.plot(traj[:,0], traj[:,1],
                        color='red',
                        linewidth=1,
                        label='Nominal trajectory')

                # optimized path
                ax.plot(opt[:,0], opt[:,1],
                        color='blue',
                        linewidth=1,
                        label='Optimized trajectory')

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                ax.legend()

                plt.colorbar(quad1, ax=ax)

                plt.title("Zoomed Optimization Deviation")

                plt.savefig(self.folder + f"/zoom_{epoch}.png",
                            bbox_inches='tight')

                plt.close(fig)

    def predict_trajectory2(self, Xsrc, Xtar,
                            samples=200,
                            step_size=0.05,
                            tol=0.01):

        device = self.Params['Device']

        Xsrc = torch.tensor(Xsrc, dtype=torch.float32, device=device)
        Xtar = torch.tensor(Xtar, dtype=torch.float32, device=device)

        current = Xsrc.clone()

        traj = [current.cpu().numpy()]

        while torch.norm(current - Xtar) > tol:

            # --------------------------------------------------
            # Sample candidate waypoints around current position
            # --------------------------------------------------

            noise = torch.randn(samples, self.dim, device=device)

            noise = noise / torch.norm(noise, dim=1, keepdim=True).clamp(min=1e-6)
            noise = noise * step_size
            goal_dir = Xtar - current
            goal_dir = goal_dir / torch.norm(goal_dir)

            noise += 0.5 * step_size * goal_dir
            candidates = current[None, :] + noise

            # keep camera height fixed (same as original planner)
            candidates[:,2] = Xsrc[2]

            # --------------------------------------------------
            # Evaluate cost using TravelTime
            # --------------------------------------------------

            goal_expand = Xtar.repeat(samples, 1)

            XP = torch.cat((candidates, goal_expand), dim=1)

            costs = self.TravelTimes(XP)

            # --------------------------------------------------
            # Choose best waypoint
            # --------------------------------------------------

            best_idx = torch.argmin(costs)

            current = candidates[best_idx]

            traj.append(current.detach().cpu().numpy())

            if len(traj) > 200:
                break

        traj.append(Xtar.cpu().numpy())

        return np.array(traj)

    def predict_trajectory2_gradient(self, Xsrc, Xtar, step_size=0.03, tol=0.03):
        Xsrc = Tensor(Xsrc)
        Xtar = Tensor(Xtar)
        XP_traj= torch.cat((Xsrc,Xtar))
        dis= torch.norm(XP_traj[:self.dim]-XP_traj[self.dim:])
        point_start = []
        point_goal = []
        iter = 0
        while dis > tol:
            gradient = self.Gradient(XP_traj[None,:].clone())
            XP_traj = XP_traj + step_size * gradient[0].cpu()
            XP_traj[2] = Xsrc[2]
            XP_traj[5] = Xsrc[2]
            dis = torch.norm(XP_traj[3:6]-XP_traj[0:3])
            point_start.append(XP_traj[0:3])
            point_goal.append(XP_traj[3:6])
            iter = iter + 1
            if iter > 200:
                break 
        
        point_goal.reverse()
        point = point_start + point_goal
        if point_start:
            traj = torch.cat(point).view(-1, 3)
            traj = torch.cat((Xsrc[None,], traj, Xtar[None,]), dim=0)
        else:
            traj = torch.cat((Xsrc[None,], Xtar[None,]), dim=0)
        # traj = np.concatenate([Xsrc[None,], traj, Xtar[None,]], axis=0)

        return traj.detach().numpy()
    

    def predict_trajectory(self, Xsrc, Xtar, step_size=0.03, tol=0.02):
        Xsrc = Tensor(Xsrc).cuda()
        Xtar = Tensor(Xtar).cuda()
        XP_traj= torch.cat((Xsrc,Xtar))
        dis= torch.norm(XP_traj[:self.dim]-XP_traj[self.dim:])
        point_start = []
        point_goal = []
        iter = 0
        while dis > tol:
            gradient = self.Gradient(XP_traj[None,:].clone())
            XP_traj = XP_traj + step_size * gradient[0]#.cpu()
            dis = torch.norm(XP_traj[3:6]-XP_traj[0:3])
            point_start.append(XP_traj[0:3])
            point_goal.append(XP_traj[3:6])
            iter = iter + 1
            if iter > 100:
                break 
        
        point_goal.reverse()
        point = point_start + point_goal
        if point_start:
            traj = torch.cat(point).view(-1, 3)
            traj = torch.cat((Xsrc[None,], traj, Xtar[None,]), dim=0)
        else:
            traj = torch.cat((Xsrc[None,], Xtar[None,]), dim=0)
        # traj = np.concatenate([Xsrc[None,], traj, Xtar[None,]], axis=0)

        return traj.detach().cpu().numpy()
    
    def policy_occ(self, current_location, height):
        # curtargetblock = self.occ_map.get_target_block()
        # if not curtargetblock:
        #     return Variable(Tensor([0, 0, height])).cpu()
        # curtargetloc = self.occ_map.get_block_center(curtargetblock)
        possible_locs = self.occ_map.get_block_centers()
        if True:
            #? find the shortest path one
            XP = np.zeros((len(possible_locs),2*self.dim))

            #?: change to small locations
            XP[:,:self.dim] = current_location
            XP[:,self.dim:self.dim+2] = possible_locs
            XP[:,self.dim+2] = height
            # print("XP", XP)
            XP = Variable(Tensor(XP)).to(self.Params['Device'])
            tt = self.TravelTimes(XP)
            tt = tt.to('cpu').data.numpy()
            small_idx = np.argmin(tt)
            curtargetloc_small = possible_locs[small_idx]  
        
        print("current location :", current_location)
        print("target location :", curtargetloc_small)
        print("block idx:", self.occ_map.block_stack[small_idx])

        #！ set a step size
        curtargetloc_small = np.concatenate([curtargetloc_small, [height]])
        curtargetloc_small_tensor = torch.from_numpy(curtargetloc_small).float()
        
        curloc_tensor = torch.from_numpy(current_location).float()

        traj_list = self.predict_trajectory2(curloc_tensor, curtargetloc_small_tensor, step_size=0.015)

        # find the largest point that is smaller than the step size
        if True:
            #? accumulate the distance to fit the step size
            step_size = 0.05
            index = 0
            curtargetloc = traj_list[-1]
            accum_dis = 0
            while accum_dis < step_size:
                curtargetloc = traj_list[index]
                accum_dis += np.linalg.norm(traj_list[index+1][:2] - traj_list[index][:2])
                index += 1
                if index >= len(traj_list)-1:
                    break
            print("accumulated distance:", accum_dis, "at traj_index:", index)
            print("Target location after gradient descent:", curtargetloc)
            print("Length of traj_list:", len(traj_list))

        #! should return a list of trajectory, and index of the end traj
        # return Variable(Tensor(curtargetloc)).cpu()
        return traj_list, index - 1
    

    def policy_goal_direction(self, current_location, height):

        possible_locs = self.occ_map.get_block_centers()
        goal = self.fixed_goal.detach().cpu().numpy()
        if len(possible_locs) == 0:
            return None, None

        current = current_location[:2]
        goal_vec = goal[:2] - current

        # normalize goal direction
        goal_norm = np.linalg.norm(goal_vec)
        if goal_norm < 1e-6:
            return None, None

        goal_dir = goal_vec / goal_norm

        scores = []

        for f in possible_locs:
            v1 = f - current
            v1_norm = np.linalg.norm(v1)

            if v1_norm < 1e-6:
                scores.append(-1e9)
                continue

            v1_dir = v1 / v1_norm

            # cosine similarity
            alignment = np.dot(v1_dir, goal_dir)
            scores.append(alignment)

        scores = np.array(scores)

        # pick best aligned frontier
        best_idx = np.argmax(scores)
        target = possible_locs[best_idx]

        target = np.concatenate([target, [height]])

        # generate trajectory to that frontier
        traj_list = self.predict_trajectory2(
            torch.tensor(current_location),
            torch.tensor(target),
            step_size=0.015
        )

        # take small step (same as before)
        step_size = 0.05
        accum_dis = 0
        index = 0

        while accum_dis < step_size and index < len(traj_list) - 1:
            accum_dis += np.linalg.norm(
                traj_list[index+1][:2] - traj_list[index][:2]
            )
            index += 1

        return traj_list, index - 1
    def policy_goal_direct(self, current_location, height):
        goal = self.fixed_goal.detach().cpu().numpy().copy()
        goal[2] = height

        traj_list = self.predict_trajectory2(
            current_location,
            goal,
            step_size=0.005
        )

        step_size = 0.05
        accum_dis = 0.0
        index = 0

        while accum_dis < step_size and index < len(traj_list) - 1:
            accum_dis += np.linalg.norm(
                traj_list[index + 1][:2] - traj_list[index][:2]
            )
            index += 1

        return traj_list, index - 1
    def reached_goal(self, current_pos, tol=0.006):
        """
        Check if current position is close enough to goal.
        """
        if torch.is_tensor(current_pos):
            current_pos = current_pos.detach().cpu().numpy()
        if torch.is_tensor(self.fixed_goal):
            goal = self.fixed_goal.detach().cpu().numpy()
        else:
            goal = self.fixed_goal

        dist = np.linalg.norm(current_pos[:2] - goal[:2])
        return dist < tol
    def compute_path_length(self, path):
        if path is None or len(path) < 2:
            return 0.0

        path = np.asarray(path)
        diffs = path[1:, :2] - path[:-1, :2]
        dists = np.linalg.norm(diffs, axis=1)
        return np.sum(dists)
    
def check_collision_with_surface_points(
    traj,
    surface_points,
    robot_radius=0.0105,
    safety_margin=0.01,
    return_details=False
):
    """
    Check if trajectory collides with obstacles using surface points.

    Args:
        traj: (N, 3) trajectory
        surface_points: (M, 3) obstacle surface points
        robot_radius: robot size
        safety_margin: extra buffer
        return_details: if True, return collision indices

    Returns:
        collision (bool)
        optional: collision indices
    """

    traj = np.asarray(traj)
    surface_points = surface_points.detach().cpu().numpy()
    surface_points = np.asarray(surface_points)

    thresh = robot_radius + safety_margin

    collision_indices = []

    for i, p in enumerate(traj):
        # 2D distance (since planner is planar)
        dists = np.linalg.norm(surface_points[:, :2] - p[:2], axis=1)

        if np.any(dists < thresh):
            collision_indices.append(i)

    collision = len(collision_indices) > 0

    if return_details:
        return collision, collision_indices

    return collision