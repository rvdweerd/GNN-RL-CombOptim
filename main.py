import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import networkx as nx
from asp import Solve_MinVertexCover_ASP
import itertools
#import networkx.algorithms.approximation
from graph import Graph
#G=nx.DiGraph()
#G.add_nodes_from([0,1,2,3,4,5])
#G.add_edges_from([(0,1),(0,4),(1,5),(1,2),(2,5),(2,3)])
#G=G.to_undirected()
#a=networkx.algorithms.approximation.vertex_cover.min_weighted_vertex_cover(G)
#A=nx.adjacency_matrix(G)
#AA = nx.convert_matrix.to_numpy_matrix(G)

G=Graph([0,1,2,3,4,5],[[0,4,1],[0,1,1],[1,5,1],[1,2,1],[2,5,1],[2,3,1]],Reflexive=False,Directed=True)
G=Graph([0,1,2,3,4,5,6,7],[[0,1,1],[0,2,1],[1,2,1],[2,3,1],[3,4,1],[4,2,1],[5,6,1],[6,7,1],[5,7,1]],Reflexive=False,Directed=True)
solutions=Solve_MinVertexCover_ASP(G)
#for sol in itertools.islice(Solve_MinCover_ASP(G),10):
#    print(sol)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

alpha = Variable(torch.tensor(0, dtype=torch.float, device=device),requires_grad=True)
delta = Variable(torch.randn(20,dtype=torch.float, device=device), requires_grad=True)
w=Variable(torch.tensor([0,0,0],dtype=torch.float,device=device),requires_grad=True)
t=Variable(torch.tensor([0,0,-500],dtype=torch.float,device=device),requires_grad=True)



lr=0.1
optimizer = optim.Adam([w,t,delta,alpha], lr=lr)
phist=[]
lhist=[]
for i in range(200):
    loss = CalcLoss(alpha)
    #loss = (1-alpha)**2
    #loss.backward()
    DoBackward(loss)
    optimizer.step()
    optimizer.zero_grad()
    phist.append(alpha.detach().item())
    lhist.append(loss)

plt.plot(phist)
plt.title('alpha')
plt.figure()
plt.plot(lhist)
plt.title('loss')
plt.show()