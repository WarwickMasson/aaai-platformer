'''
Plot runs.
'''
import matplotlib.pyplot as plt
from learn import load, SHIFT_VECTOR, SCALE_VECTOR, fourier_basis, STATE_DIM, FixedSarsaAgent
import numpy as np
import simulator
import scipy.stats as t
from mpl_toolkits.mplot3d import axes3d

VALUE_STEPS = 100

def average_return(returns):
    ''' Compute the average return for a set of episodes. '''
    total = 0
    values = np.zeros(returns.shape)
    for i in range(returns.size):
        total += returns[i]
        values[i] = total / (i + 1)
    return values

def plot_return(agent, returns, data=None):
    ''' Plot return over time. '''
    plt.plot(returns, '-' + agent.colour, label=agent.legend)
    interval = 2000
    if data != None:
        for i in range(returns.size/interval):
            plt.errorbar(1+i*interval, returns[i*interval],
                yerr=t.sem(data[:, i*interval]), fmt='-'+agent.colour)
    plt.axis([0, returns.size, 0.0, 1.0])
    plt.xlabel('Episodes')
    plt.title('Average Return')
    plt.ylabel('Average Return')

def plot_run(agent_class, run):
    ''' Plot a single run. '''
    agent = load(agent_class, run)
    returns = average_return(np.load(agent.filename + '.npy'))
    plot_return(agent, returns)
    plt.savefig(agent.filename + '.png', bbox_inches='tight')

def plot_tdiffs(agent_class, runs=20):
    ''' Plot tdiff for the runs. '''
    agent = load(agent_class, 1)
    tdiffs = 0*np.array(agent.tdiffs)
    for run in range(1, runs + 1):
        agent = load(agent_class, run)
        tdiffs += np.array(agent.tdiffs) / runs
    plt.plot(tdiffs, '-' + agent.colour, label=agent.legend)
    plt.xlabel('Episodes')
    plt.title('Average Delta')
    plt.ylabel('Delta')
    plt.savefig('./runs/delta.png', bbox_inches='tight')

def plot_return_agents(agents, max_runs, runs=50):
    ''' Plot all the average returns for all agents. '''
    plt.clf()
    for agent_class in agents:
        returns = np.zeros((max_runs,))
        data = np.zeros((runs, max_runs))
        for run in range(1, runs + 1):
            agent = load(agent_class, run)
            ret = np.load(agent.filename + '.npy')
            ret = average_return(ret[:max_runs])
            returns += ret / runs
            data[run-1, :] = ret
        plot_return(agent, returns, data)
    plt.legend(loc='upper left')
    plt.savefig('./runs/return', bbox_inches='tight')

PLOT_EPISODES = 50

def plot_episode(agent_class, run):
    ''' Plot an example run. '''
    agent = load(agent_class, run)
    sims = []
    for _ in range(PLOT_EPISODES):
        sim = simulator.Simulator()
        agent.run_episode(sim)
        sims.append(sim)
    import interface
    interface.Interface().draw_episode(sims, 'after')

def plot_value_function(agent_class, run, i):
    ''' Plot the value functions for run i. '''
    plt.clf()
    agent = load(agent_class, run)
    state0 = simulator.Simulator().get_state()
    values, qval1, qval2 = [], [], []
    min_range = -SHIFT_VECTOR[i]
    max_range = SCALE_VECTOR[i]
    variables = []
    for j in range(VALUE_STEPS):
        var = max_range*(1.*j / VALUE_STEPS) + min_range
        state0[i] = var
        values.append(agent.value_function(state0))
        feat = agent.action_features[0](state0)
        qval1.append(agent.action_weights[0].dot(feat))
        qval2.append(agent.action_weights[1].dot(feat))
        variables.append(var)
    max_val = max(max(qval1), max(qval2), min(values))
    min_val = min(min(qval1), min(qval2), min(values))
    plt.plot(variables, values, '-b', label='$V(s)$')
    plt.plot(variables, qval1, '-r', label='$Q(s, a_1)$')
    plt.plot(variables, qval2, '-g', label='$Q(s, a_2)$')
    plt.axis([min_range, max_range, min_val, max_val])
    plt.legend(loc='lower right')
    plt.xlabel(str(i))
    plt.ylabel('$V$')
    plt.savefig('./runs/' + agent.name + '/value_functions/s' + str(i), bbox_inches='tight')

def plot_all_vfs(agent, run):
    ''' Plot all the value functions. '''
    for i in range(STATE_DIM):
        plot_value_function(agent, run, i)

def plot_x_dx(agent_class, run):
    ''' Plot the value function over x, dx. '''
    plt.clf()
    agent = load(agent_class, run)
    fig = plt.figure()
    state = simulator.Simulator().get_state()
    plot = fig.add_subplot(111, projection='3d')
    plot.set_xlabel('x')
    plot.set_ylabel('dx')
    plot.set_zlabel('Action-Value')
    xxrange = np.arange(0, 1000, 10.0)
    yyrange = np.arange(0, 200, 10.0)
    xgrid, ygrid = np.meshgrid(xxrange, yyrange)
    get_state = lambda x, dx: np.append(np.array([x, dx]), state[2:])
    function2 = lambda x, dx: agent.action_weights[0].dot(fourier_basis(get_state(x, dx)))
    function3 = lambda x, dx: agent.action_weights[1].dot(fourier_basis(get_state(x, dx)))
    functions = [function2, function3]
    colours = [[1, 0, 0]]
    for col, func in zip(colours, functions):
        zarray = np.array([func(x, dx) for x, dx in zip(np.ravel(xgrid), np.ravel(ygrid))])
        zgrid = zarray.reshape(xgrid.shape)
        print col
        plot.plot_surface(xgrid, ygrid, zgrid, color=col)
    plt.savefig('./runs/' + agent.name + '/value_functions/xdx', bbox_inches='tight')

def plot_cooling():
    ''' Plot the effect of different cooling parameters. '''
    plt.clf()
    fig = plt.figure()
    plot = fig.add_subplot(111, projection='3d')
    plot.set_ylabel('Cooling')
    plot.set_xlabel('Episodes')
    plot.set_zlabel('Temperature')
    crange = np.arange(0.99, 1.0, 0.0001)
    eprange = np.arange(0, 2000, 2.0)
    xgrid, ygrid = np.meshgrid(eprange, crange)
    func = lambda cool, ep: cool**ep
    zarray = np.array([func(cool, ep) for ep, cool in zip(np.ravel(xgrid), np.ravel(ygrid))])
    zgrid = zarray.reshape(xgrid.shape)
    plot.plot_surface(xgrid, ygrid, zgrid, color=[1, 0, 0, 0.9])
    plt.savefig('./runs/cooling.png', bbox_inches='tight')
    plt.clf()
    eprange2 = range(5000)
    values = [func(FixedSarsaAgent.cooling, ep) for ep in eprange2]
    plt.plot(eprange2, values, '-r')
    plt.xlabel('Episodes')
    plt.ylabel('Temperature')
    plt.savefig('./runs/coolsetting.png', bbox_inches='tight')
