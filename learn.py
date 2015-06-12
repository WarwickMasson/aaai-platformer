'''
This file implements learning agents for the goal domain.
'''
import numpy as np
import pickle
from numpy.linalg import norm
from simulator import Simulator, MAX_WIDTH, MAX_GAP, HEIGHT_DIFF
from simulator import MAX_PLATWIDTH, MAX_SPEED, Enemy
from random import choice
from util import to_matrix
import cma

def softmax(values):
    ''' Returns the softmax weighting of a set of values. '''
    maxval = max(values)
    values = [np.exp(value - maxval) for value in values]
    total = sum(values)
    return [value / total for value in values]

def weighted_selection(values):
    ''' Select an index with probabilities given by values. '''
    rand = np.random.rand()
    for index, value in enumerate(values):
        if rand <= value:
            return index
        rand -= value
    return 0

FOURIER_DIM = 7
def generate_coefficients(coeffs, vector = np.zeros((13,)), depth = 0, count = 0):
    ''' Generate all coefficient vectors. '''
    if depth == vector.size or count == 2:
        coeffs.append(vector)
    else:
        for j in range(FOURIER_DIM):
            new_vector = np.copy(vector)
            new_vector[depth] = np.pi * j
            generate_coefficients(coeffs, new_vector, depth+1, count + (j > 0))

SHIFT_VECTOR = np.array([0.0, 0.0, Enemy.size[0], 20.0, Enemy.size[0], 20.0, 0.0, 0.0, 2*HEIGHT_DIFF, 0.0, 0.0, 2*HEIGHT_DIFF, 0.0]) 
SCALE_VECTOR = np.array([MAX_WIDTH, MAX_SPEED, MAX_WIDTH, 40.0, MAX_WIDTH, 40.0,
MAX_PLATWIDTH, MAX_WIDTH, 4*HEIGHT_DIFF, MAX_PLATWIDTH, MAX_WIDTH, 4*HEIGHT_DIFF, MAX_WIDTH])
COEFFS = []
generate_coefficients(COEFFS)
BASIS_COUNT = len(COEFFS)
print BASIS_COUNT
COEFF_SCALE = np.ones(BASIS_COUNT)
for i in range(1, BASIS_COUNT):
    COEFF_SCALE[i] = norm(COEFFS[i])

def scale_state(state):
    ''' Scale state variables between 0 and 1. '''
    new_state = np.copy(state)
    scaled = (new_state + SHIFT_VECTOR) / SCALE_VECTOR
    for i in range(scaled.size):
        if not 0 <= scaled[i] <= 1:
            print i, scaled[i]
            assert(1 == 0)
    return scaled

def fourier_basis(state):
    ''' Defines a fourier basis function. '''
    basis = np.zeros((BASIS_COUNT,))
    scaled = scale_state(state)
    for i, coeff in enumerate(COEFFS):
        basis[i] = np.cos(coeff.dot(scaled))
    return basis

def enemy_features(state):
    return np.array([1, state[0], state[1], state[7], state[8]])

def gap_features(state):
    return np.array([1, state[0], state[1], state[2], state[3], state[4], state[5], state[6]])

class Agent:
    '''
    Implements an agent with a parameterized or weighted policy.
    '''

    action_count = 2
    temperature = 1.0
    variance = 0.1
    gamma = 0.9
    parameter_features = [gap_features, enemy_features]
    parameter_weights = [
        np.array([1, 0, 0, 0, 0, 0, 0, 0]),
        np.array([50, 0, 0, 0, 0])]

    def __init__(self):
        self.action_weights = []

    def run_episode(self, simulator = None):
        ''' Run a single episode for a maximum number of steps. '''
        if simulator == None:
            simulator = Simulator()
        state = simulator.get_state()
        states = [state]
        rewards = []
        actions = []
        acts = []
        end_ep = False
        while not end_ep:
            act = self.action_policy(state)
            action = self.policy(state, act)
            state, reward, end_ep, _ = simulator.take_action(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            acts.append(act)
        return states, actions, rewards, acts

    def evaluate_policy(self, runs):
        ''' Evaluate the current policy. '''
        average_reward = 0
        for _ in range(runs):
            rewards = self.run_episode()[2]
            average_reward += sum(rewards) / runs
        return average_reward

    def policy(self, state, action = None):
        ''' Policy selects an action based on its internal policies. '''
        if action == None:
            action = self.action_policy(state)
        parameters = self.parameter_policy(state, action)
        action_names = ['run', 'jump']
        return (action_names[action], parameters)

    def action_prob(self, state):
        ''' Computes the probability of selecting each action. '''
        values = []
        for i in range(self.action_count):
            features = self.action_features[i](state)
            val = self.action_weights[i].T.dot(features)
            values.append(val / self.temperature)
        return softmax(values)

    def action_policy(self, state):
        ''' Selects an action based on action probabilities. '''
        values = self.action_prob(state)
        return weighted_selection(values)

    def parameter_policy(self, state, action):
        ''' Computes the parameters for the given action. '''
        features = self.parameter_features[action](state)
        weights = self.parameter_weights[action]
        mean = weights.T.dot(features)
        return np.random.normal(mean, self.variance)

    def get_parameters(self):
        ''' Returns all the parameters in a vector. '''
        parameters = np.zeros((0,))
        for action in range(self.action_count):
            parameters = np.append(parameters, self.action_weights[action])
            parameters = np.append(parameters, self.parameter_weights[action])
        return parameters

    def set_parameters(self, parameters):
        ''' Set the parameters using a vector. '''
        index = 0
        for action in range(self.action_count):
            size = self.action_weights[action].size
            self.action_weights[action] = parameters[index: index+size]
            index += size
            rows = self.parameter_weights[action].size
            self.parameter_weights[action] = parameters[index: index+rows]
            index += rows

    def log_action_gradient(self, state, action, selection):
        ''' Returns the log gradient for action,
            given the state and the selection used. '''
        features = self.action_features[action](state)
        prob = self.action_prob(state)[action]
        if action == selection:
            return (1 - prob)*features / self.temperature
        else:
            return - prob * features / self.temperature

    def log_parameter_gradient(self, state, action, value):
        ''' Returns the log gradient for the parameter,
            given the state and the value. '''
        features = self.parameter_features[action](state)
        mean = self.parameter_weights[action].dot(features)
        grad = (value - mean) * features / self.variance
        return grad

    def log_gradient(self, state, action, value):
        ''' Returns the log gradient for the entire policy. '''
        grad = np.zeros((0,))
        for i in range(self.action_count):
            action_grad = self.log_action_gradient(state, i, action)
            grad = np.append(grad, action_grad)
            rows = self.parameter_weights[i].size
            if i == action:
                parameter_grad = self.log_parameter_gradient(state, i, value)
                grad = np.append(grad, parameter_grad)
            else:
                grad = np.append(grad, np.zeros((rows,)))
        return grad

    def update(self):
        ''' Perform one learning update. '''
        pass

    def learn(self, steps):
        ''' Learn for the given number of update steps. '''
        returns = []
        total = 0.0
        for step in range(steps):
            rets = self.update()
            returns.append(sum(rets))
            total += sum(rets)
            print 'Step:', step, sum(rets), total / (step + 1)
        return returns

class FixedSarsaAgent(Agent):
    ''' A fixed parameter weight gradient-descent SARSA agent. '''

    name = 'fixedsarsa'
    colour = 'b'
    legend = 'Fixed Sarsa'
    alpha = 0.1
    lmb = 0.0
    action_features = []

    def __init__(self):
        ''' Initialize coeffs. '''
        self.action_weights = []
        for _ in range(self.action_count):
            self.action_weights.append(np.zeros((BASIS_COUNT,)))
            self.action_features.append(fourier_basis)

    def update(self):
        ''' Learn for a single episode. '''
        simulator = Simulator()
        state = simulator.get_state()
        act = self.action_policy(state)
        feat = self.action_features[act](state)
        end_episode = False
        rewards = []
        traces = []
        for i in range(self.action_count):
            traces.append(np.zeros((BASIS_COUNT,)))
        while not end_episode:
            action = self.policy(state, act)
            state, reward, end_episode, step = simulator.take_action(action)
            new_act = self.action_policy(state)
            new_feat = self.action_features[new_act](state)
            rewards.append(reward)
            delta = reward + (self.gamma ** step) * self.action_weights[new_act].dot(new_feat) - self.action_weights[act].dot(feat)
            for i in range(self.action_count):
                traces[i] *= self.lmb * self.gamma
            traces[act] += feat
            for i in range(self.action_count):
                self.action_weights[i] += self.alpha * delta * traces[i] / COEFF_SCALE
            act = new_act
            feat = new_feat
        return rewards

class CmaesAgent(Agent):
    ''' Defines a CMA-ES agent. '''

    colour = 'r'
    legend = 'CMA-ES'
    name = 'cmaes'
    runs = 5
    sigma = 0.1

    def objective_function(self, container, parameters):
        ''' Defines a simple objective function for direct optimization. '''
        self.set_parameters(parameters)
        total = 0
        for _ in range(self.runs):
            reward = self.evaluate_policy(1)
            total -= reward / self.runs
            container.append(reward)
        return total

    def learn(self, _):
        ''' Learn until convergence. '''
        returns = []
        function = lambda parameters: self.objective_function(returns, parameters)
        res = cma.fmin(function, self.get_parameters(), self.sigma)
        self.set_parameters(res[5])
        return returns

class AlternatingAgent(FixedSarsaAgent):
    ''' Alternates learning using Sarsa and Cmaes. '''

    colour = 'b'
    legend = 'Alternating Optimization'
    name = 'ao'
    qsteps = 1000
    runs = 5
    sigma = 0.1

    def objective_function(self, container, parameters):
        ''' Defines a simple objective function for direct optimization. '''
        self.set_parameters(parameters)
        total = 0
        for _ in range(self.runs):
            reward = self.evaluate_policy(1)
            total -= reward / self.runs
            container.append(reward)
        return total

    def get_parameters(self):
        ''' Get the parameter weights. '''
        parameters = np.zeros((0,))
        for action in range(self.action_count):
            parameters = np.append(parameters, self.parameter_weights[action])
        return parameters

    def set_parameters(self, parameters):
        ''' Set the parameters using a vector. '''
        index = 0
        for action in range(self.action_count):
            rows = self.parameter_weights[action].size
            self.parameter_weights[action] = parameters[index: index+rows]
            index += rows

    def learn(self, steps):
        ''' Learn for a given number of steps. '''
        returns = []
        function = lambda parameters: self.objective_function(returns, parameters)
        for step in range(steps):
            agent = FixedSarsaAgent()
            agent.action_weights = self.action_weights
            agent.parameter_weights = self.parameter_weights
            rets = agent.learn(self.qsteps)
            returns.append(sum(rets))
            res = cma.fmin(function, self.get_parameters(), self.sigma)
            self.set_parameters(res[5])
        return returns

class QpamdpAgent(FixedSarsaAgent):
    ''' Defines an agen to optimize H(theta) using eNAC. '''

    relearn = 100
    runs = 100
    name = 'qpamdp'
    legend = 'Q-PAMDP'
    colour = 'g'
    beta = 1.0

    def get_parameters(self):
        ''' Get the parameter weights. '''
        parameters = np.zeros((0,))
        for action in range(self.action_count):
            parameters = np.append(parameters, self.parameter_weights[action])
        return parameters

    def set_parameters(self, parameters):
        ''' Set the parameters using a vector. '''
        index = 0
        for action in range(self.action_count):
            rows = self.parameter_weights[action].size
            self.parameter_weights[action] = parameters[index: index+rows]
            index += rows

    def log_gradient(self, state, action, value):
        ''' Returns the log gradient for the entire policy. '''
        grad = np.zeros((0,))
        for i in range(self.action_count):
            rows = self.parameter_weights[i].size
            if i == action:
                parameter_grad = self.log_parameter_gradient(state, i, value)
                grad = np.append(grad, parameter_grad)
            else:
                grad = np.append(grad, np.zeros((rows,)))
        return grad

    def value_function(self, state):
        value = 0
        action_prob = self.action_prob(state)
        for act in range(self.action_count):
            feat = self.action_features[act](state)
            value += action_prob[act]* self.action_weights[act].dot(feat)
        return value

    def enac_gradient(self):
        ''' Compute the episodic NAC gradient. '''
        returns = np.zeros((self.runs, 1))
        phi = lambda state: np.array([1, state[2], state[3], state[4], state[5], state[6], state[2]**2, state[3]**2,
            state[4]**2, state[5]**2, state[6]**2])
        param_size = self.get_parameters().size
        psi = np.zeros((self.runs, param_size+11))
        for run in range(self.runs):
            states, actions, rewards, acts = self.run_episode()
            returns[run, 0] = sum(rewards)
            log_grad = np.zeros((param_size,))
            for state, act, action in zip(states, acts, actions):
                val = action[1]
                log_grad += self.log_gradient(state, act, val)
            psi[run, :] = np.append(log_grad, phi(states[0]))
        omega_v = np.linalg.pinv(psi).dot(returns)
        grad = omega_v[0:param_size, 0]
        return grad, returns

    def parameter_update(self):
        ''' Perform a single gradient update. '''
        grad, returns = self.enac_gradient()
        if norm(grad) > 0:
            grad /= norm(grad)
        self.set_parameters(self.get_parameters() + self.beta * grad)
        return returns

    def learn(self, steps):
        ''' Learn for a given number of steps. '''
        returns = []
        for step in range(500):
            new_ret = self.update()
            returns.append(sum(new_ret))
        for step in range(steps):
            new_ret = self.parameter_update()
            print sum(new_ret) / self.runs
            returns.extend(new_ret)
            for update in range(self.relearn):
                new_ret = self.update()
                print sum(new_ret)
                returns.append(sum(new_ret))
            print step
        return returns

class EnacAoAgent(QpamdpAgent):
    ''' Defines an alternating agent using eNAC. '''

    name = 'enacao'
    legend = 'AO'
    colour = 'b'

    def learn(self, steps):
        ''' Learn for a given number of steps. '''
        returns = []
        for step in range(steps):
            for i in range(500):
                new_ret = self.update()
                print i, sum(new_ret)
                returns.append(sum(new_ret))
            for i in range(0):
                new_ret = self.parameter_update()
		returns.extend(new_ret)
                print step, i, sum(new_ret) / len(new_ret)
        return returns

class EnacAgent(QpamdpAgent):
    ''' Defines an agent to optimize J(theta, omega) using eNAC. '''

    name = 'enac'
    legend = 'eNAC'
    colour = 'r'

    def get_parameters(self):
        ''' Returns all the parameters in a vector. '''
        parameters = np.zeros((0,))
        for action in range(self.action_count):
            parameters = np.append(parameters, self.action_weights[action])
            parameters = np.append(parameters, self.parameter_weights[action])
        return parameters

    def set_parameters(self, parameters):
        ''' Set the parameters using a vector. '''
        index = 0
        for action in range(self.action_count):
            size = self.action_weights[action].size
            self.action_weights[action] = parameters[index: index+size]
            index += size
            rows = self.parameter_weights[action].size
            self.parameter_weights[action] = parameters[index: index+rows]
            index += rows

    def log_gradient(self, state, action, value):
        ''' Returns the log gradient for the entire policy. '''
        grad = np.zeros((0,))
        for i in range(self.action_count):
            action_grad = self.log_action_gradient(state, i, action)
            grad = np.append(grad, action_grad)
            rows = self.parameter_weights[i].size
            if i == action:
                parameter_grad = self.log_parameter_gradient(state, i, value)
                grad = np.append(grad, parameter_grad)
            else:
                grad = np.append(grad, np.zeros((rows,)))
        return grad

    def learn(self, steps):
        ''' Learn for a given number of steps. '''
        returns = []
        for step in range(steps):
            new_ret = self.parameter_update()
            print new_ret
            returns.append(sum(new_ret))
            print step
        return returns

def determine_variance(agent, steps, runs = 1):
    ''' Determine the variance of parameterized policy agent. '''
    rewards = []
    for _ in range(steps):
        reward = agent.evaluate_policy(runs)
        rewards.append(reward)
        print reward
    mean = sum(rewards) / steps
    variance = 0
    for reward in rewards:
        variance += (reward - mean)**2 / steps
    print
    print 'Mean:', mean
    print 'Variance:', variance

def save_run(agent_class, steps, run):
    ''' Save a single run. '''
    agent = agent_class()
    returns = np.array(agent.learn(steps))
    np.save('./runs/'+agent.name+'/'+str(run), returns)
    with file('./runs/'+agent.name+'/'+str(run)+'.obj', 'w') as file_handle:
        pickle.dump(agent, file_handle)

def extend_run(agent_class, steps, run):
    ''' Extend an existing run for a given number of steps. '''
    agent = None
    with file('./runs/'+agent_class.name +'/'+str(run)+'.obj', 'r') as file_handle:
        agent = pickle.load(file_handle)
        run_name = './runs/'+agent.name+'/'+str(run)+'.npy'
        returns = np.load(run_name)
        returns = np.append(returns, agent.learn(steps))
        np.save(run_name, returns)
    if agent != None:
        with file('./runs/'+agent_class.name +'/'+str(run)+'.obj', 'w') as file_handle:
            pickle.dump(agent, file_handle)

def random_sample():
    print 0, sum(QpamdpAgent().learn(0)) / 500
    for i in range(1, 10):
        agent = QpamdpAgent()
        params = agent.get_parameters()
        params += 2*np.random.randn(params.size)
        agent.set_parameters(params)
        rets = agent.learn(0)
        print i, sum(rets) / 500
        print agent.get_parameters()
