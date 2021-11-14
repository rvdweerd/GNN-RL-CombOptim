import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from asp import Solve_MinVertexCover_ASP
from graph import Graph
from utils import *
from collections import namedtuple
from qnet import QNet

seed_everything(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

G=Graph([0,1,2,3,4,5],[[0,4,1],[0,1,1],[1,5,1],[1,2,1],[2,5,1],[2,3,1]],Reflexive=False,Directed=True)
solutions=Solve_MinVertexCover_ASP(G)
State = namedtuple('State',('W', 'node', 'partial_solution'))



# GPU / batch optimized embedding
model=QNet(5,T=4).to(device)
#coords, W_np = get_graph_mat(n=10)

W=torch.tensor(G.W,dtype=torch.float32,device=device)
xv=torch.zeros((1,G.num_nodes,1)).to(device)
Ws=W.unsqueeze(0)

y=model(xv,Ws)
k=0
