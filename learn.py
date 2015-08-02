'''
This file implements learning agents for the goal domain.
'''
import numpy as np
import pickle
from numpy.linalg import norm
from simulator import Simulator, MAX_WIDTH, ENEMY_SPEED
from simulator import MAX_DX, Enemy, Player

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

def formatf(value):
    ''' Format a float to 5 decimal places. '''
    return '{0:.5f}'.format(value)

def formatd(value):
    ''' Format an integer to 6 places. '''
    return '{0:6d}'.format(int(value))

def format_array(values):
    ''' Format a float array. '''
    return [formatf(value) for value in values]

FOURIER_DIM = 6
STATE_DIM = Simulator().get_state().size
COUPLING = STATE_DIM - 2
def generate_coefficients(coeffs, vector, depth=0, count=0):
    ''' Generate all coefficient vectors. '''
    if depth == STATE_DIM or count == COUPLING:
        coeffs.append(vector)
    else:
        for j in range(FOURIER_DIM):
            new_vector = np.copy(vector)
            new_vector[depth] = np.pi * j
            generate_coefficients(coeffs, new_vector, depth+1, count + (j > 0))

def get_coeffs():
    ''' Compute coeffs, scale, count. '''
    coeffs = []
    generate_coefficients(coeffs, np.zeros((STATE_DIM,)))
    count = len(coeffs)
    scale = np.ones(count)
    for i in range(1, count):
        scale[i] = norm(coeffs[i])
    return coeffs, scale, count

SHIFT_VECTOR = np.array([Player.size[0], 0.0, 0.0,
    ENEMY_SPEED])
SCALE_VECTOR = np.array([MAX_WIDTH + Player.size[0], MAX_DX,
    MAX_WIDTH, 2*ENEMY_SPEED])
COEFFS, COEFF_SCALE, BASIS_COUNT = get_coeffs()
print "Basis Functions:", BASIS_COUNT
INITIAL_RUN = 1.0
INITIAL_HOP = 100.0
INITIAL_LEAP = 500.0
CHECK_SCALE = False

def scale_state(state):
    ''' Scale state variables between 0 and 1. '''
    new_state = np.copy(state)
    scaled = (new_state + SHIFT_VECTOR) / SCALE_VECTOR
    if CHECK_SCALE:
        for i in range(scaled.size):
            if not 0 <= scaled[i] <= 1:
                print i, scaled[i], new_state[i]
                assert 0 <= scaled[i] <= 1
    return scaled

def fourier_basis(state):
    ''' Defines a fourier basis function. '''
    basis = np.zeros((BASIS_COUNT,))
    scaled = scale_state(state)
    for i, coeff in enumerate(COEFFS):
        basis[i] = np.cos(coeff.dot(scaled))
    return basis

def polynomial_basis(state):
    ''' Defines a polynomial basis using the current COEFFS. '''
    basis = np.zeros((BASIS_COUNT,))
    scaled = scale_state(state)
    for i, coeff in enumerate(COEFFS):
        basis[i] = coeff.dot(scaled)
    basis[0] = 1.0
    return basis

def param_features(state):
    ''' Defines a simple linear set of state variables. '''
    array = np.ones(state.size + 1)
    array[1:] = scale_state(state)
    return array

def initial_features(state):
    ''' Computes the initial features phi(s_0) for enac. '''
    return np.array([1])

def load(agent_class, run):
    ''' Load the given class. '''
    agent = agent_class(run)
    file_handle = file(agent.filename + '.obj', 'r')
    agent = pickle.load(file_handle)
    return agent

def save(agent):
    ''' Save the agent. '''
    file_handle = file(agent.filename + '.obj', 'w')
    pickle.dump(agent, file_handle)

class FixedSarsaAgent:
    '''
    Implements a fixed parameter weight gradient-descent SARSA agent.
    '''

    name = 'fixedsarsa'
    legend = 'Fixed Sarsa'
    colour = 'r'
    action_count = 3
    lmb = 0.5
    gamma = 0.9
    cooling = 0.996
    variances = [0.001, 0.1, 0.1]
    action_names = ['run', 'hop', 'leap']
    parameter_features = [param_features, param_features, param_features]
    action_features = [fourier_basis, fourier_basis, fourier_basis]

    def __init__(self, run):
        self.run = run
        self.action_weights = []
        self.filename = 'runs/' + self.name +'/'+ str(run)
        self.parameter_weights = [
            INITIAL_RUN*np.eye(STATE_DIM + 1, 1)[:, 0],
            INITIAL_HOP*np.eye(STATE_DIM + 1, 1)[:, 0],
            INITIAL_LEAP*np.eye(STATE_DIM + 1, 1)[:, 0]]
        for _ in range(self.action_count):
            self.action_weights.append(np.zeros((BASIS_COUNT,)))
        self.steps = 0.0
        self.tdiff = 0.0
        self.tdiffs = []
        self.total = 0.0
        self.episodes = 0.0
        self.returns = []
        self.alpha = 1.0
        self.temperature = 1.0

    def run_episode(self, simulator=None):
        ''' Run a single episode for a maximum number of steps. '''
        if simulator == None:
            simulator = Simulator()
        state = simulator.get_state()
        states = [state]
        rewards = []
        actions = []
        end_ep = False
        act = self.action_policy(state)
        acts = [act]
        while not end_ep:
            action = self.policy(state, act)
            new_state, reward, end_ep, _ = simulator.take_action(action)
            new_act = self.action_policy(new_state)
            delta = reward - self.state_quality(state, act)
            if not end_ep:
                delta += self.gamma * self.state_quality(new_state, new_act)
            self.tdiff += abs(delta)
            self.steps += 1.0
            state = new_state
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            act = new_act
            acts.append(act)
        self.tdiffs.append(self.tdiff / self.steps)
        self.episodes += 1
        self.total += sum(rewards)
        self.returns.append(sum(rewards))
        return states, actions, rewards, acts

    def discount(self, rewards):
        ''' Computes the discounted sum of rewards. '''
        return sum([(self.gamma**t)*reward for t, reward in enumerate(rewards)])

    def value_function(self, state):
        ''' Computes V(s) = E_a[pi(s,a)Q(s,a)] '''
        value = 0
        action_prob = self.action_prob(state)
        for act in range(self.action_count):
            value += action_prob[act] * self.state_quality(state, act)
        return value

    def state_quality(self, state, act):
        ''' Returns Q(s, a). '''
        feat = self.action_features[act](state)
        return self.feat_quality(feat, act)

    def feat_quality(self, feat, act):
        ''' Returns w_a * phi. '''
        return self.action_weights[act].dot(feat)

    def evaluate_policy(self, runs):
        ''' Evaluate the current policy. '''
        average_reward = 0
        for _ in range(runs):
            rewards = self.run_episode()[2]
            average_reward += sum(rewards) / runs
        return average_reward

    def follow_action(self, act):
        ''' Computes the expected return after taking action a. '''
        sim = Simulator()
        action = self.policy(sim.get_state(), act)
        reward, end = sim.take_action(action)[1:3]
        if end:
            return reward
        else:
            rewards = self.run_episode(sim)[2]
            return reward + self.gamma * self.discount(rewards)

    def compare_value_function(self, runs):
        ''' Compares the value function to the expected rewards. '''
        ret = 0.0
        rets = [0]*self.action_count
        quality = [0]*self.action_count
        sim = Simulator()
        state = sim.get_state()
        vf0 = self.value_function(state)
        for j in range(self.action_count):
            quality[j] = self.state_quality(state, j)
        for i in range(1, runs + 1):
            ret += self.discount(self.run_episode()[2]) / runs
            for j in range(self.action_count):
                rets[j] += self.follow_action(j) / runs
            print 'Step: ', formatd(i), 'V(s0): ', formatf(vf0), 'R: ', formatf(ret * runs / i)
        print "V: ", formatf(vf0)
        print "R:", formatf(ret)
        print "Q:", [formatf(qual) for qual in quality]
        print "RQ:", [formatf(retn) for retn in rets]

    def load_runs(self):
        ''' Load the saved results for the agent. '''
        return np.load(self.filename + '.npy')

    def save_runs(self, returns):
        ''' Save the returns. '''
        np.save(self.filename + '.npy', np.array(returns))

    def policy(self, state, action=None):
        ''' Policy selects an action based on its internal policies. '''
        if action == None:
            action = self.action_policy(state)
        parameters = self.parameter_policy(state, action)
        return (self.action_names[action], parameters)

    def action_prob(self, state):
        ''' Computes the probability of selecting each action. '''
        values = [self.state_quality(state, i) for i in range(self.action_count)]
        if self.temperature == 0.0:
            prob = [0]*self.action_count
            prob[np.argmax(values)] = 1.0
        else:
            prob = softmax([val / self.temperature for val in values])
        return prob

    def action_policy(self, state):
        ''' Selects an action based on action probabilities. '''
        values = self.action_prob(state)
        return weighted_selection(values)

    def parameter_policy(self, state, action):
        ''' Computes the parameters for the given action. '''
        features = self.parameter_features[action](state)
        weights = self.parameter_weights[action]
        mean = weights.dot(features)
        if self.variances[action] == 0.0:
            return mean
        else:
            return np.random.normal(mean, self.variances[action])

    def update(self):
        ''' Learn for a single episode. '''
        simulator = Simulator()
        state = simulator.get_state()
        act = self.action_policy(state)
        feat = self.action_features[act](state)
        end_episode = False
        rewards = []
        traces = []
        for _ in range(self.action_count):
            traces.append(np.zeros((BASIS_COUNT,)))
        while not end_episode:
            action = self.policy(state, act)
            state, reward, end_episode, _ = simulator.take_action(action)
            new_act = self.action_policy(state)
            new_feat = self.action_features[new_act](state)
            rewards.append(reward)
            delta = reward - self.feat_quality(feat, act)
            if not end_episode:
                delta += self.gamma * self.feat_quality(new_feat, new_act)
            self.tdiff += abs(delta)
            self.steps += 1.0
            for i in range(self.action_count):
                traces[i] *= self.lmb * self.gamma
            traces[act] += feat
            alpha_bound = self.gamma * traces[new_act].dot(new_feat) - traces[act].dot(feat)
            self.alpha = min(self.alpha, 1.0 / abs(alpha_bound))
            for i in range(self.action_count):
                self.action_weights[i] += self.alpha * delta * traces[i] / COEFF_SCALE
            act = new_act
            feat = new_feat
        self.episodes += 1
        total_ret = sum(rewards)
        self.total += total_ret
        self.temperature *= self.cooling
        self.returns.append(total_ret)
        self.tdiffs.append(self.tdiff / self.steps)
        av_ret = self.total / self.episodes
        av_diff = self.tdiff / self.steps
        print 'Step:', formatd(self.episodes), 'r:', formatf(total_ret), 'R:', formatf(av_ret), 'Delta:', formatf(av_diff)
        return rewards

    def learn(self, steps):
        ''' Learn for the given number of update steps. '''
        for step in range(steps):
            rets = self.update()
        return self.returns

class HardcodedAgent(FixedSarsaAgent):
    ''' A hard-coded fixed deterministic policy agent. '''

    name = 'hardcoded'
    legend = 'Hardcoded Agent'
    colour = 'k'

    def action_policy(self, state):
        ''' Selects an action. '''
        if state[0] == 0:
            return 0
        else:
            return 1

class QpamdpAgent(FixedSarsaAgent):
    ''' Defines an agent to optimize H(theta) using eNAC. '''

    relearn = 50
    runs = 50
    name = 'qpamdp'
    legend = 'Q-PAMDP'
    colour = 'g'
    beta = 0.1
    qsteps = 2000
    opt_omega = False

    def get_parameters(self):
        ''' Returns all the parameters in a vector. '''
        parameters = np.zeros((0,))
        for action in range(self.action_count):
            if self.opt_omega:
                parameters = np.append(parameters, self.action_weights[action])
            parameters = np.append(parameters, self.parameter_weights[action])
        return parameters

    def set_parameters(self, parameters):
        ''' Set the parameters using a vector. '''
        index = 0
        for action in range(self.action_count):
            if self.opt_omega:
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
        grad = (value - mean) * features / self.variances[action]
        return grad

    def log_gradient(self, state, action, value):
        ''' Returns the log gradient for the entire policy. '''
        grad = np.zeros((0,))
        for i in range(self.action_count):
            if self.opt_omega:
                action_grad = self.log_action_gradient(state, i, action)
                grad = np.append(grad, action_grad)
            rows = self.parameter_weights[i].size
            if i == action:
                parameter_grad = self.log_parameter_gradient(state, i, value)
                grad = np.append(grad, parameter_grad)
            else:
                grad = np.append(grad, np.zeros((rows,)))
        return grad

    def enac_gradient(self):
        ''' Compute the episodic NAC gradient. '''
        returns = np.zeros((self.runs, 1))
        param_size = self.get_parameters().size
        feat_size = initial_features(np.zeros((STATE_DIM,))).size
        psi = np.zeros((self.runs, param_size+feat_size))
        for run in range(self.runs):
            states, actions, rewards, acts = self.run_episode()
            returns[run, 0] = sum(rewards)
            log_grad = np.zeros((param_size,))
            for state, act, action in zip(states, acts, actions):
                log_grad += self.log_gradient(state, act, action[1])
            psi[run, :] = np.append(log_grad, initial_features(states[0]))
            av_ret = self.total / self.episodes
            av_diff = self.tdiff / self.steps
            print 'Step:', formatd(self.episodes), 'r:', formatf(sum(rewards)), 'R:', formatf(av_ret), 'Delta:', formatf(av_diff)
        grad = np.linalg.pinv(psi).dot(returns)[0:param_size, 0]
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
        for step in range(self.qsteps):
            new_ret = self.update()
        updates = int((steps - self.qsteps) / (self.runs + self.relearn))
        for step in range(updates):
            new_ret = self.parameter_update()
            for _ in range(self.relearn):
                new_ret = self.update()
        return self.returns

class EnacAoAgent(QpamdpAgent):
    ''' Defines an alternating agent using eNAC. '''

    name = 'enacao'
    legend = 'AO'
    colour = 'b'
    gradsteps = 100

    def learn(self, steps):
        ''' Learn for a given number of steps. '''
        updates = int(steps / (self.qsteps + self.gradsteps * self.runs))
        for _ in range(updates):
            for i in range(self.qsteps):
                new_ret = self.update()
            for i in range(self.gradsteps):
                new_ret = self.parameter_update()
        return self.returns

class EnacAgent(QpamdpAgent):
    ''' Defines an agent to optimize J(theta, omega) using eNAC. '''

    name = 'enac'
    legend = 'eNAC'
    colour = 'r'
    opt_omega = True

    def learn(self, steps):
        ''' Learn for a given number of steps. '''
        updates = int(steps / self.runs)
        for step in range(updates):
            new_ret = self.parameter_update()
        return self.returns

def determine_variance(agent, steps, runs=1):
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
    agent = agent_class(run)
    returns = agent.learn(steps)
    agent.save_runs(returns)
    save(agent)

def extend_run(agent_class, steps, run):
    ''' Extend an existing run for a given number of steps. '''
    agent = load(agent_class, run) 
    returns = agent.load_runs()
    returns = np.append(returns, agent.learn(steps))
    agent.save_runs(returns)
    save(agent)

def random_sample(runs):
    ''' Randomly tests parameters around the current parameters. '''
    print 0, sum(QpamdpAgent().learn(0)) / QpamdpAgent.qsteps
    for i in range(1, runs):
        agent = QpamdpAgent()
        params = agent.get_parameters()
        params += 2*np.random.randn(params.size)
        agent.set_parameters(params)
        rets = agent.learn(0)
        print i, sum(rets) / QpamdpAgent.qsteps
        print agent.get_parameters()

def gradient_variance(runs):
    ''' Compute the variance of the gradient estimate. '''
    agent = load(QpamdpAgent, 999)
    grads = []
    for _ in range(runs):
        grad = agent.enac_gradient()[0]
        grad /= norm(grad)
        grads.append(grad)
        print grad
    print
    mean = np.mean(grads, 0)
    print 'Mean:'
    print mean
    var = np.var(grads, 0)
    print 'Var:'
    print var
    print 'Relative Var:'
    print mean / var
    print 'Norm Var:', norm(var)
