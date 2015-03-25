'''
Used for executing non-fixed commands.
'''
import learn
import plot
from learn import QpamdpAgent
from simulator import Simulator
from interface import Interface
import pickle

plot_agents = [learn.QpamdpAgent, learn.CmaesAgent, learn.AlternatingAgent]
plot_num = 30000
plot.plot_return_agents(plot_agents, plot_num)
plot.plot_goals_agents(plot_agents, plot_num)
'''
agent = QpamdpAgent
with file('./runs/'+agent.name+'/'+str(1)+'.obj', 'r') as file_handle:
    agent = pickle.load(file_handle)
    sim = Simulator()
    agent.run_episode(sim)
    interface = Interface(sim)
    interface.plot_episode('episode')
'''


'''
run = 1000
steps = 300
agent = learn.QpamdpAgent
learn.save_run(agent, steps, run)
plot.plot_run(agent, run)
plot.plot_goals(agent, run)
'''
