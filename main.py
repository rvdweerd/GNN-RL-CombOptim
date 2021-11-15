import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from asp import Solve_MinVertexCover_ASP
from graph import Graph
from utils import *
from collections import namedtuple
from qnet import QNet, QFunction

SEED = 10  # A seed for the random number generator
# Graph
NR_NODES = 6  # Number of nodes N
EMBEDDING_DIMENSIONS = 8#5  # Embedding dimension D
EMBEDDING_ITERATIONS_T = 4  # Number of embedding iterations T
#G=Graph([0,1,2,3,4,5],[[0,4,1],[0,1,1],[1,5,1],[1,2,1],[2,5,1],[2,3,1]],Reflexive=False,Directed=True)
#solutions=Solve_MinVertexCover_ASP(G)
# Learning
NR_EPISODES = 10001
MEMORY_CAPACITY = 5000
N_STEP_QL = 2  # Number of steps (n) in n-step Q-learning to wait before computing target reward estimate
BATCH_SIZE = 16
GAMMA = 0.9
INIT_LR = 1e-3
LR_DECAY_RATE = 0.99998#1. - 2e-5  # learning rate decay
MIN_EPSILON = 0.1
EPSILON_0 = 0.5
EPSILON_DECAY_RATE = 6e-4  # epsilon decay
FOLDER_NAME = './models'  # where to checkpoint the best models

seed_everything(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

State = namedtuple('State',('G', 'partial_solution', 'num_nodes', 'min_VC', 'candidates'))

def Test():
    G=Graph([0,1,2,3,4,5],[[0,4,1],[0,1,1],[1,5,1],[1,2,1],[2,5,1],[2,3,1]],Reflexive=False,Directed=True)
    solutions=Solve_MinVertexCover_ASP(G)
    # GPU / batch optimized embedding
    model=QNet(5,T=4).to(device)
    #coords, W_np = get_graph_mat(n=10)
    W=torch.tensor(G.W,dtype=torch.float32,device=device)
    xv=torch.zeros((1,G.num_nodes,1)).to(device)
    Ws=W.unsqueeze(0)
    y=model(xv,Ws)
    k=0

def init_model(fname=None):
    """ Create a new model. If fname is defined, load the model from the specified file.
    """
    Q_net = QNet(EMBEDDING_DIMENSIONS, T=EMBEDDING_ITERATIONS_T).to(device)
    optimizer = optim.Adam(Q_net.parameters(), lr=INIT_LR)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_RATE)
    
    if fname is not None:
        checkpoint = torch.load(fname)
        Q_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    Q_func = QFunction(Q_net, optimizer, lr_scheduler)
    return Q_func, Q_net, optimizer, lr_scheduler

def checkpoint_model(model, optimizer, lr_scheduler, loss, 
                     episode, avg_length):
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)
    
    fname = os.path.join(FOLDER_NAME, 'ep_{}'.format(episode))
    fname += '_length_{}'.format(avg_length)
    fname += '.tar'
    
    torch.save({
        'episode': episode,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'loss': loss,
        'avg_length': avg_length
    }, fname)

# Create module, optimizer, LR scheduler, and Q-function
Q_func, Q_net, optimizer, lr_scheduler = init_model()

# Create memory
memory = Memory(MEMORY_CAPACITY)

# Storing metrics about training:
found_solutions = dict()  # episode --> (W, solution)
losses = []
path_length_ratios = []

# keep track of mean ratio of estimated MVC / real MVC
current_min_mean_ratio = float('inf')

for episode in range(NR_EPISODES):
    # sample a new random graph
    #coords, W_np = get_graph_mat(n=NR_NODES)
    G, solutions = get_graph(n=NR_NODES)
    true_minVC=len(solutions[0])
    W = torch.tensor(G.W, dtype=torch.float32, requires_grad=False, device=device)
    
    # current partial solution - a list of node index
    solution = []#random.randint(0, NR_NODES-1)]
    remaining_candidates = [i for i in range(NR_NODES) if i not in solution]
    
    # current state (tuple and tensor)
    current_state = State(partial_solution=solution, G=G, num_nodes=NR_NODES, min_VC=true_minVC, candidates=remaining_candidates)
    current_state_tsr = state2tensor(current_state)
    
    # Keep track of some variables for insertion in replay memory:
    states = [current_state]
    states_tsrs = [current_state_tsr]  # we also keep the state tensors here (for efficiency)
    rewards = []
    actions = []
    
    # current value of epsilon
    epsilon = max(MIN_EPSILON, EPSILON_0*((1-EPSILON_DECAY_RATE)**episode))
    
    nr_explores = 0
    t = -1
    while not is_state_final(current_state,G):
        t += 1  # time step of this episode
        
        if epsilon >= random.random():
            # explore
            next_node = get_next_neighbor_random(current_state)
            nr_explores += 1
            if episode % 50 == 0:
                print('Ep {} explore | current sol: {} | sol: {}'.format(episode, solution, solutions),'nextnode',next_node)
        else:
            # exploit
            next_node, est_reward = Q_func.get_best_action(current_state_tsr, current_state)
            if episode % 50 == 0:
                print('Ep {} exploit | current sol: {} / next est reward: {} | sol: {}'.format(episode, solution, est_reward,solutions),'nextnode',next_node)
        
        next_solution = solution + [next_node]
        next_remaining_candidates = [i for i in range(NR_NODES) if i not in next_solution]

        # reward observed for taking this step        
        reward = -1.
        next_state = State(partial_solution=next_solution, G=G, num_nodes=NR_NODES, min_VC=true_minVC, candidates=next_remaining_candidates)
        next_state_tsr = state2tensor(next_state)
        
        # store rewards and states obtained along this episode:
        states.append(next_state)
        states_tsrs.append(next_state_tsr)
        rewards.append(reward)
        actions.append(next_node)
        
        # store our experience in memory, using n-step Q-learning:
        if len(solution) >= N_STEP_QL:
            memory.remember(Experience(state=states[-N_STEP_QL],
                                       state_tsr=states_tsrs[-N_STEP_QL],
                                       action=actions[-N_STEP_QL],
                                       reward=sum(rewards[-N_STEP_QL:]),
                                       next_state=next_state,
                                       next_state_tsr=next_state_tsr))
            
        if is_state_final(next_state,G):
            for n in range(1, N_STEP_QL):
                memory.remember(Experience(state=states[-n],
                                           state_tsr=states_tsrs[-n], 
                                           action=actions[-n], 
                                           reward=sum(rewards[-n:]), 
                                           next_state=next_state,
                                           next_state_tsr=next_state_tsr))
        
        # update state and current solution
        current_state = next_state
        current_state_tsr = next_state_tsr
        solution = next_solution
        
        # take a gradient step
        loss = None
        if len(memory) >= BATCH_SIZE and len(memory) >= 400:
            experiences = memory.sample_batch(BATCH_SIZE)
            
            batch_states_tsrs = [e.state_tsr for e in experiences]
            batch_Ws = [torch.tensor(e.state.G.W,dtype=torch.float32,device=device) for e in experiences]
            batch_actions = [e.action for e in experiences]
            batch_targets = []
            
            for i, experience in enumerate(experiences):
                target = experience.reward
                if not is_state_final(experience.next_state,G):
                    _, best_reward = Q_func.get_best_action(experience.next_state_tsr, 
                                                            experience.next_state)
                    target += GAMMA * best_reward
                batch_targets.append(target)
                
            # print('batch targets: {}'.format(batch_targets))
            loss = Q_func.batch_update(batch_states_tsrs, batch_Ws, batch_actions, batch_targets)
            losses.append(loss)
            
            """ Save model when we reach a new low average path length
            """
            #med_length = np.median(path_length_ratios[-100:])
            mean_ratio = int(np.mean(path_length_ratios[-100:])*100)/100
            if mean_ratio < current_min_mean_ratio:
                current_min_mean_ratio = mean_ratio
                checkpoint_model(Q_net, optimizer, lr_scheduler, loss, episode, mean_ratio)
                
    length = len(solution)
    optimal_length = len(solutions[0])
    path_length_ratios.append(length/optimal_length)

    if episode % 10 == 0:
        print('Ep %d. Loss = %.3f / median length = %.3f / last = %.4f / epsilon = %.4f / lr = %.4f' % (
            episode, (-1 if loss is None else loss), np.mean(path_length_ratios[-50:]), length/optimal_length, epsilon,
            Q_func.optimizer.param_groups[0]['lr']))
        #print(path_length_ratios[-50:])
        found_solutions[episode] = (W.clone(), [n for n in solution])




def _moving_avg(x, N=10):
    return np.convolve(np.array(x), np.ones((N,))/N, mode='valid')

plt.figure(figsize=(8,5))
plt.semilogy(_moving_avg(losses, 100))
plt.ylabel('loss')
plt.xlabel('training iteration')

plt.figure(figsize=(8,5))
plt.plot(_moving_avg(path_length_ratios, 100))
plt.ylabel('average length')
plt.xlabel('episode')

plt.show()

