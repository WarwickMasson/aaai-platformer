'''
Plot runs.
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from learn import *
import numpy as np
import pickle
import simulator
import scipy.stats as t

def average_return(returns):
    ''' Compute the average return for a set of episodes. '''
    total = 0
    values = np.zeros(returns.shape)
    for i in range(returns.size):
        total += returns[i]
        values[i] = total / (i + 1)
    return values

def sliding_window(returns):
    count = 0.0
    prob = np.zeros(returns.shape)
    window = 1000.0
    for i in range(returns.size):
        if i > window:
            if returns[i-window] == 0:
                count -= 1.0
            denom = window
        else:
            denom = (i + 1)
        if returns[i] == 0:
            count += 1.0
        prob[i] = count / denom
    return prob

def plot_return(agent, returns, data = None):
    ''' Plot return over time. '''
    plt.plot(returns, '-' + agent.colour, label = agent.legend)
    n = returns.size
    interval = 2000
    if data != None:
        for i in range(n/interval):
            plt.errorbar(1+i*interval, returns[i*interval], 
                yerr = t.sem(data[:, i*interval]), fmt= '-'+agent.colour)
    plt.xlabel('Episodes')
    plt.title('Average Return')
    plt.ylabel('Average Return')

def plot_run(agent, run):
    ''' Plot a single run. '''
    item_name = './runs/'+agent.name+'/'+str(run)
    returns = average_return(np.load(item_name + '.npy'))
    plot_return(agent, returns)
    plt.savefig(item_name, bbox_inches='tight')

def plot_average_goals(agent, returns, data = None):
    ''' Plot the average goals over time. '''
    prob = average_return(returns)
    n = returns.size
    interval = 2000
    if data != None:
        for i in range(data.shape[0]):
            data[i, :] = average_return(data[i, :])
        for i in range(n/interval):
            plt.errorbar(1+i*interval, prob[i*interval], 
                 yerr = t.sem(data[:, i*interval]), fmt= '-'+agent.colour)
    plt.plot(prob, '-' + agent.colour, label = agent.legend)
    plt.title('Goal Probability')
    plt.axis([0, prob.size, 0.0, 0.6])
    plt.xlabel('Episodes')
    plt.ylabel('Goal Probability')

def goal_count(returns):
    ''' Boolean array for returns based on goals. '''
    return 1. * np.array([(val == 50.) for val in returns])

def plot_goals(agent, run):
    ''' Plot a single run. '''
    item_name = './runs/'+agent.name+'/'+str(run)
    returns = goal_count(np.load(item_name + '.npy'))
    plot_average_goals(agent, returns)
    plt.savefig(item_name+'_gb', bbox_inches='tight')

def plot_return_agents(agents, max_runs, runs = 50):
    ''' Plot all the average returns for all agents. '''
    plt.clf()
    for agent in agents:
        returns = np.zeros((max_runs,))
        data = np.zeros((runs, max_runs))
        for run in range(1, runs + 1):
            ret = np.load('./runs/' + agent.name + '/' + str(run) + '.npy')
            ret = average_return(ret[:max_runs])
            returns += ret / runs
            data[run-1, :] = ret
        plot_return(agent, returns, data)
    plt.legend(loc = 'upper left')
    plt.savefig('./runs/return', bbox_inches='tight')

def plot_goals_agents(agents, max_runs, runs = 50):
    ''' Plot all the goals for all agents. '''
    plt.clf()
    for agent in agents:
        returns = np.zeros((max_runs,))
        data = np.zeros((runs, max_runs))
        for run in range(1, runs + 1):
            ret = np.load('./runs/'+agent.name + '/' + str(run) + '.npy')
            ret = goal_count(ret[:max_runs])
            returns += ret / runs
            data[run-1, :] = ret
        plot_average_goals(agent, returns, data)
    plt.legend(loc = 'upper left')
    plt.savefig('./runs/goals', bbox_inches='tight')

def plot_action_weights(weights, name, run):
    ''' Plot the action weights as a function of x,y. '''
    plt.clf()
    fig = plt.figure()
    plot = fig.add_subplot(111, projection = '3d')
    xxrange = np.arange(0, 30, 0.5)
    yyrange = np.arange(-15, 15, 0.5)
    xgrid, ygrid = np.meshgrid(xxrange, yyrange)
    colours = [[1,0,0],[0,1,0],[0,0,1]]
    for weight, colour in zip(weights, colours):
        function = lambda x,y: weight.dot(fourier_basis(np.array([x, y, 0, 0, 0])))
        zarray = np.array([function(x, y) for x,y in zip(np.ravel(xgrid), np.ravel(ygrid))])
        zgrid = zarray.reshape(xgrid.shape)
        plot.plot_surface(xgrid, ygrid, zgrid, color = colour)
    plot.set_xlabel('x')
    plot.set_ylabel('y')
    plot.set_zlabel('Action-Value')
    plt.savefig('./runs/'+name+'/'+str(run)+'_weight', bbox_inches='tight')

def plot_episode(agent, run):
    ''' Plot an example run. '''
    with file('./runs/'+agent.name+'/'+str(run)+'.obj', 'r') as file_handle:
        agent = pickle.load(file_handle)
        sim = simulator.Simulator()
        import interface
        agent.run_episode(sim)
        interface.Interface().draw_episode(sim, 'after')

def plot_policy(agent, run):
    ''' Plot policies. '''
    with file('./runs/'+agent.name+'/'+str(run)+'.obj', 'r') as file_handle:
        agent = pickle.load(file_handle)
        plot_action_weights(agent.action_weights, agent.name, run)

def print_policy(agent, run):
    with file('./runs/'+agent.name+'/'+str(run)+'.obj', 'r') as file_handle:
        agent = pickle.load(file_handle)
        print agent.action_weights
        print agent.parameter_weights
        print agent.alpha

def plot_value_functions(agent, run):
    with file('./runs/'+agent.name+'/'+str(run)+'.obj', 'r') as file_handle:
        agent = pickle.load(file_handle)
        state0 = simulator.Simulator().get_state()
        values, qval1, qval2 = [], [], []
        val = agent.action_weights[0]
        for x in range(1000):
            state0[0] = x
            values.append(agent.value_function(state0))
            feat = agent.action_features[0](state0)
            qval1.append(agent.action_weights[0].dot(feat))
            qval2.append(agent.action_weights[1].dot(feat))
        plt.plot(values, '-b', label = '$V(s)$')
        plt.plot(qval1, '-r', label = '$Q(s, a_1)$')
        plt.plot(qval2, '-g', label = '$Q(s, a_2)$')
        plt.axis([0, 1000, -2, 2])
        plt.legend(loc = 'lower right')
        plt.xlabel('$x$')
        plt.ylabel('$V$')
        plt.savefig('./runs/value_function', bbox_inches='tight')
