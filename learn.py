'''
This file implements learning agents for the goal domain.
'''
import numpy as np
import pickle
from numpy.linalg import norm
from simulator import Simulator, MAX_WIDTH, MAX_TIME, MAX_GAP, HEIGHT_DIFF
from simulator import MAX_PLATWIDTH, MAX_SPEED, MAX_SPIKES
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

def generate_coefficients(coeffs, vector = np.zeros((8,)), depth = 0):
    ''' Generate all coefficient vectors. '''
    if depth == 2:
        coeffs.append(vector)
    else:
        for j in range(3):
            new_vector = np.copy(vector)
            new_vector[depth] = np.pi * j
            generate_coefficients(coeffs, new_vector, depth+1)

FOURIER_DIM = 3
SCALE_VECTOR = np.array([MAX_WIDTH, MAX_SPEED, MAX_PLATWIDTH, MAX_GAP, MAX_SPIKES, HEIGHT_DIFF, MAX_PLATWIDTH, MAX_TIME])
SHIFT_VECTOR = np.array([0, 0, 0, 0, 0, HEIGHT_DIFF, 0, 0])
COEFFS = []
generate_coefficients(COEFFS)
BASIS_COUNT = len(COEFFS)
COEFF_SCALE = np.ones(BASIS_COUNT)
for i in range(1, BASIS_COUNT):
    COEFF_SCALE[i] = norm(COEFFS[i])

def scale_state(state):
    ''' Scale state variables between 0 and 1. '''
    new_state = np.copy(state)
    return (new_state + SHIFT_VECTOR) / SCALE_VECTOR

def fourier_basis(state):
    ''' Defines a fourier basis function. '''
    basis = np.zeros((BASIS_COUNT,))
    scaled = scale_state(state)
    for i, coeff in enumerate(COEFFS):
        basis[i] = np.cos(coeff.dot(scaled))
    return basis

def run_features(state):
    return np.append(state, [1])

def jump_features(state):
    return np.append(state, [1])

class Agent:
    '''
    Implements an agent with a parameterized or weighted policy.
    '''

    action_count = 2
    temperature = 1.0
    variance = 0.1
    alpha = 0.1
    gamma = 0.9
    num = 100
    parameter_features = [run_features, jump_features]
    action_weights = [[],[],[]]
    parameter_weights = [
        np.array([100, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([3000, 0, 0, 0, 0, 0, 0, 0, 0])]

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
            state, reward, end_ep = simulator.take_action(action)
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
            cols = self.parameter_weights[action].shape[1]
            for col in range(cols):
                parameters = np.append(parameters, self.parameter_weights[action][:, col])
        return parameters

    def set_parameters(self, parameters):
        ''' Set the parameters using a vector. '''
        index = 0
        for action in range(self.action_count):
            size = self.action_weights[action].size
            self.action_weights[action] = parameters[index: index+size]
            index += size
            rows, cols = self.parameter_weights[action].shape
            for col in range(cols):
                self.parameter_weights[action][:, col] = parameters[index: index+rows]
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

    def log_parameter_gradient(self, state, action, value, col):
        ''' Returns the log gradient for the parameter,
            given the state and the col of values. '''
        features = self.parameter_features[action](state)
        mean = self.parameter_weights[action][:, col].dot(features)
        grad = (value - mean) * features / self.variance
        return grad

    def log_gradient(self, state, action, value):
        ''' Returns the log gradient for the entire policy. '''
        grad = np.zeros((0,))
        for i in range(self.action_count):
            action_grad = self.log_action_gradient(state, i, action)
            grad = np.append(grad, action_grad)
            rows, cols = self.parameter_weights[i].shape
            if i == action:
                for col in range(cols):
                    parameter_grad = self.log_parameter_gradient(state, i, value[col], col)
                    grad = np.append(grad, parameter_grad)
            else:
                grad = np.append(grad, np.zeros((rows*cols,)))
        return grad

    def update(self):
        ''' Perform one learning update. '''
        pass

    def learn(self, steps):
        ''' Learn for the given number of update steps. '''
        returns = []
        total = 0
        for step in range(steps):
            rets = self.update()
            self.alpha *= (self.num + step) / (self.num + step + 1.0)
            returns.extend(rets)
            total += sum(rets)
            print 'Step:', step, sum(rets), total / (step + 1)
        return returns

class SarsaAgent(Agent):
    ''' A discretized parameter weight gradient-descent SARSA agent. '''

    alpha = 0.1
    name = 'sarsa'
    legend = 'Sarsa(0)'
    colour = 'r'
    num = 1000

    def __init__(self):
        ''' Initialize the Sarsa Agent. '''
        Agent.__init__(self)
        self.action_count = 0
        self.actions = []
        self.action_weights = []
        self.action_features = []
        self.act_map = {}
        resolution = 6
        xvals = range(0, PITCH_LENGTH/2, resolution)
        yvals = range(-PITCH_WIDTH/2, PITCH_WIDTH/2, resolution)
        for xval in xvals:
            for yval in yvals:
                self.action_weights.append(np.zeros((5,)))
                self.action_features.append(position_features)
                action = ('kickto', (xval, yval))
                self.actions.append(action)
                self.act_map[action] = self.action_count
                self.action_count += 1
        for gval in range(-7, 8, 2):
            self.action_weights.append(np.zeros((5,)))
            self.action_features.append(position_features)
            action = ('shootgoal', (gval,))
            self.actions.append(action)
            self.act_map[action] = self.action_count
            self.action_count += 1

    def policy(self, state):
        ''' Define a discrete policy. '''
        act = self.action_policy(state)
        return self.actions[act]

    def update(self):
        ''' Learn for a single episode. '''
        simulator = Simulator()
        state = simulator.get_state()
        action = self.policy(state)
        act = self.act_map[action]
        feat = self.action_features[act](state)
        end_episode = False
        while not end_episode:
            new_state, reward, end_episode = simulator.take_action(action)
            new_action = self.policy(new_state)
            new_act = self.act_map[new_action]
            new_feat = self.action_features[new_act](new_state)
            delta = reward + self.gamma * self.action_weights[new_act].dot(new_feat) - self.action_weights[act].dot(feat)
            self.action_weights[act] += self.alpha * delta * feat
            state = new_state
            act = new_act
            feat = new_feat
            action = new_action
        return [reward]

class FixedSarsaAgent(Agent):
    ''' A fixed parameter weight gradient-descent SARSA agent. '''

    name = 'fixedsarsa'
    colour = 'b'
    legend = 'Fixed Sarsa'
    alpha = 0.005
    lmb = 0.0
    action_features = [fourier_basis, fourier_basis, fourier_basis]

    def __init__(self):
        ''' Initialize coeffs. '''
        for i in range(3):
            self.action_weights[i] = np.zeros((BASIS_COUNT,))

    def update(self):
        ''' Learn for a single episode. '''
        simulator = Simulator()
        state = simulator.get_state()
        act = self.action_policy(state)
        feat = self.action_features[act](state)
        end_episode = False
        rewards = []
        traces = [
            np.zeros((BASIS_COUNT,)),
            np.zeros((BASIS_COUNT,)),
            np.zeros((BASIS_COUNT,))]
        while not end_episode:
            action = self.policy(state, act)
            state, reward, end_episode = simulator.take_action(action)
            new_act = self.action_policy(state)
            new_feat = self.action_features[new_act](state)
            rewards.append(reward)
            delta = reward + self.gamma * self.action_weights[new_act].dot(new_feat) - self.action_weights[act].dot(feat)
            for i in range(3):
                traces[i] *= self.lmb * self.gamma
            traces[act] += feat
            for i in range(3):
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
            cols = self.parameter_weights[action].shape[1]
            for col in range(cols):
                parameters = np.append(parameters, self.parameter_weights[action][:, col])
        return parameters

    def set_parameters(self, parameters):
        ''' Set the parameters using a vector. '''
        index = 0
        for action in range(self.action_count):
            rows, cols = self.parameter_weights[action].shape
            for col in range(cols):
                self.parameter_weights[action][:, col] = parameters[index: index+rows]
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
            returns.extend(rets)
            res = cma.fmin(function, self.get_parameters(), self.sigma)
            self.set_parameters(res[5])
        return returns

class QpamdpAgent(FixedSarsaAgent):
    ''' Defines an agen to optimize H(theta) using eNAC. '''

    relearn = 100
    runs = 50
    name = 'qpamdp'
    legend = 'Q-PAMDP'
    colour = 'g'
    beta = 0.2
    jval = 0.0
    jrate = 0.1
    num = 100
    num2 = 5000

    def get_parameters(self):
        ''' Get the parameter weights. '''
        parameters = np.zeros((0,))
        for action in range(self.action_count):
            cols = self.parameter_weights[action].shape[1]
            for col in range(cols):
                parameters = np.append(parameters, self.parameter_weights[action][:, col])
        return parameters

    def set_parameters(self, parameters):
        ''' Set the parameters using a vector. '''
        index = 0
        for action in range(self.action_count):
            rows, cols = self.parameter_weights[action].shape
            for col in range(cols):
                self.parameter_weights[action][:, col] = parameters[index: index+rows]
                index += rows

    def log_gradient(self, state, action, value):
        ''' Returns the log gradient for the entire policy. '''
        grad = np.zeros((0,))
        for i in range(self.action_count):
            rows, cols = self.parameter_weights[i].shape
            if i == action:
                for col in range(cols):
                    parameter_grad = self.log_parameter_gradient(state, i, value[col], col)
                    grad = np.append(grad, parameter_grad)
            else:
                grad = np.append(grad, np.zeros((rows*cols,)))
        return grad

    def value_function(self, state):
        value = 0
        action_prob = self.action_prob(state)
        for act in range(3):
            feat = self.action_features[act](state)
            value += action_prob[act]* self.action_weights[act].dot(feat)
        return value

    def enac_gradient(self):
        ''' Compute the episodic NAC gradient. '''
        returns = np.zeros((self.runs, 1))
        phi = lambda state: np.array([1, state[1], state[1]**2])
        param_size = self.get_parameters().size
        psi = np.zeros((self.runs, param_size+3))
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

    def delta_gradient(self):
        ''' Compute the delta gradient. '''
        param_size = self.get_parameters().size
        psi = np.zeros((param_size, 0))
        delta = np.zeros((0, 1))
        returns = []
        for run in range(self.runs):
            states, actions, rewards, acts = self.run_episode()
            t = 0
            for state, action, reward, act in zip(states, actions, rewards, acts):
                val = action[1]
                psi = np.append(psi, to_matrix(self.log_gradient(state, act, val)), 1)
                delta_value = reward + self.gamma * self.value_function(states[t+1]) - self.value_function(state) - self.jval
                delta = np.append(delta, [delta_value])
                t += 1
            returns.append(sum(rewards))
        grad = np.linalg.pinv(psi.T).dot(delta)
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
        for step in range(2000):
            new_ret = self.update()
            self.alpha *= (self.num + step) / (self.num + step + 1.0)
            print new_ret
            returns.extend(new_ret)
            self.jval = (1 - self.jrate)*self.jval + self.jrate*sum(new_ret)
        for step in range(steps):
            new_ret = self.parameter_update()
            print new_ret
            returns.extend(new_ret)
            for update in range(self.relearn):
                new_ret = self.update()
                print new_ret
                returns.extend(new_ret)
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
        for step in range(2000):
            new_ret = self.update()
            self.alpha *= (self.num + step) / (self.num + step + 1.0)
            print new_ret
            returns.extend(new_ret)
            self.jval = (1 - self.jrate)*self.jval + self.jrate*sum(new_ret)
        for step in range(steps):
            for i in range(1000):
                new_ret = self.parameter_update()
		returns.extend(new_ret)
            self.beta *= (self.num2 + step) / (self.num2 + step + 1.0)
            print new_ret
            for update in range(2000):
                new_ret = self.update()
                print new_ret
                returns.extend(new_ret)
                self.jval = (1 - self.jrate)*self.jval + self.jrate*sum(new_ret)
            self.alpha *= (self.num2 + step) / (self.num2 + step + 1.0)
            print step
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
            cols = self.parameter_weights[action].shape[1]
            for col in range(cols):
                parameters = np.append(parameters, self.parameter_weights[action][:, col])
        return parameters

    def set_parameters(self, parameters):
        ''' Set the parameters using a vector. '''
        index = 0
        for action in range(self.action_count):
            size = self.action_weights[action].size
            self.action_weights[action] = parameters[index: index+size]
            index += size
            rows, cols = self.parameter_weights[action].shape
            for col in range(cols):
                self.parameter_weights[action][:, col] = parameters[index: index+rows]
                index += rows

    def log_gradient(self, state, action, value):
        ''' Returns the log gradient for the entire policy. '''
        grad = np.zeros((0,))
        for i in range(self.action_count):
            action_grad = self.log_action_gradient(state, i, action)
            grad = np.append(grad, action_grad)
            rows, cols = self.parameter_weights[i].shape
            if i == action:
                for col in range(cols):
                    parameter_grad = self.log_parameter_gradient(state, i, value[col], col)
                    grad = np.append(grad, parameter_grad)
            else:
                grad = np.append(grad, np.zeros((rows*cols,)))
        return grad

    def learn(self, steps):
        ''' Learn for a given number of steps. '''
        returns = []
        for step in range(steps):
            new_ret = self.parameter_update()
            print new_ret
            returns.extend(new_ret)
            self.alpha *= (self.num2 + step) / (self.num2 + step + 1.0)
            print step
        return returns

def determine_variance(steps, runs = 1):
    ''' Determine the variance of parameterized policy agent. '''
    agent = Agent()
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
