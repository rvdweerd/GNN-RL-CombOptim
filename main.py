import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from asp import Solve_MinVertexCover_ASP
from graph import Graph

G=Graph([0,1,2,3,4,5],[[0,4,1],[0,1,1],[1,5,1],[1,2,1],[2,5,1],[2,3,1]],Reflexive=False,Directed=True)
solutions=Solve_MinVertexCover_ASP(G)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

p=5
T=4
theta1 = Variable(torch.randn((p,1), dtype=torch.float, device=device),requires_grad=True)
theta2 = Variable(torch.randn((p,p), dtype=torch.float, device=device),requires_grad=True)
theta3 = Variable(torch.randn((p,p), dtype=torch.float, device=device),requires_grad=True)
theta4 = Variable(torch.randn((p,1), dtype=torch.float, device=device),requires_grad=True)
theta5 = Variable(torch.randn((2*p,1), dtype=torch.float, device=device),requires_grad=True)
theta6 = Variable(torch.randn((p,p), dtype=torch.float, device=device),requires_grad=True)
theta7 = Variable(torch.randn((p,p), dtype=torch.float, device=device),requires_grad=True)

n=G.num_nodes
S=set()
in_set = torch.zeros(n)
mu = torch.zeros((p,n))
for t in range(T):
    for node in G.node_list:
        F1=theta1*in_set[node]

        F2=theta2@mu.sum(dim=1,keepdim=True)

        F3=torch.zeros((p,1))
        for e in G.out_edges[node]:
            F3+=nn.ReLU()(theta4*e[1])
        F3=theta3@F3
        mu[:,node]+= nn.ReLU()(F1+F2+F3).squeeze()
    print(mu)

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