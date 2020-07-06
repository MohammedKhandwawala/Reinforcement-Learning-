from gym import Env
from gym.envs.registration import register
from gym.utils import seeding
from gym import spaces
import numpy as np

class gridworld(Env):
	def __init__(self):
		self.m = 12
		self.n = 12
		# self.grid = np.zeros((self.m,self.n))
		self.stateSpace = [i for i in range(self.m*self.n)]
		self.stateSpacePlus = [i for i in range(self.m*self.n)]
		self.actionSpace = {'N':-self.m , 'S':self.m , 'W':-1, 'E':1}
		self.possibleActions	 = ['N','S','E','W']
		self.agentPosition = np.random.choice([12*5,12*6,12*10,12*11])
		self.category = 1 #can 1 2 or 3

	def setInitParam(self, category,goalstate):
		self.category = category #1,2 or 3
		self.goalstate = goalstate
		self.stateSpace.remove(goalstate)

	def isTerminalState(self,state):
		return state in self.stateSpacePlus and state not in self.stateSpace

	def getAgentRowAndColumn(self):
		 x = self.agentPosition // self.m
		 y = self.agentPosition % self.n
		 return x,y


	def _seed(self, seed=None):
			self.np_random, seed = seeding.np_random(seed)
			return [seed]

	def setState(self , state):
		x,y = self.getAgentRowAndColumn()
		# self.grid[x][y] = 0
		self.agentPosition = state
		x, y =self.getAgentRowAndColumn()
		# self.grid[x][y] = 1

	def offGridMove(self,newState , oldState):
		if newState not in self.stateSpacePlus:
			return True
		elif oldState % self.m == 0 and newState % self.m == self.m -1 :
			return True
		elif oldState % self.m == self.m -1 and newState % self.m == 0:
			return True
		return False

	def _step(self, action):
		x,y = self.getAgentRowAndColumn()
		rand_action = set(self.actionSpace)
		rand_action.remove(action)
		rand_action = list(rand_action)
		probs = [0.9 , 0.1/3 , 0.1/3, 0.1/3]	
		
		resultingState = self.agentPosition + np.random.choice([self.actionSpace[action],self.actionSpace[rand_action[0]],self.actionSpace[rand_action[1]],self.actionSpace[rand_action[2]]],1,p=probs)[0]

		if self.category != 3 :
 			resultingState = resultingState + np.random.choice(range(2),1,[0.5,0.5])[0]

		if not self.offGridMove(resultingState,self.agentPosition):
			self.setState(resultingState)
			reward = self.getReward(resultingState)
			return resultingState, reward , self.isTerminalState(self.agentPosition), None
		else:	
			reward = self.getReward(self.agentPosition)
			return self.agentPosition , reward , self.isTerminalState(self.agentPosition), None

	def _reset(self):
		self.agentPosition = np.random.choice([12*5,12*6,12*10,12*11])
		# self.grid = np.zeros((self.m, self.n))
		return self.agentPosition

	def getReward(self,state):
		re = np.zeros((12,12))
		re[2][3:9] = -1
		re[3][3:9] = -1
		re[4][3:9] = -1
		re[5][3:9] = -1
		re[6][3:9] = -1
		re[7][3:8] = -1
		re[8][3:8] = -1
		re[3][4:8] = -2
		re[4][4:8] = -2
		re[5][4:8] = -2
		re[6][4:7] = -2
		re[7][4:7] = -2
		re[4][5] = -3
		re[4][6] = -3
		re[5][5] = -3
		re[6][5] = -3
		if self.category == 1:
			re[0][11] = 10
		elif self.category == 2:
			re[2][9] = 10
		elif self.category == 3:
			re[6][7] = 10
		x = state // self.m
		y = state % self.n
		return re[x][y]

	def render(self):
		print('---------------------------------')
		# for row in self.grid:
			# for col in row:
				# if col == 0:
					# print('-',end="\t")
				# elif col == 1:
					# print('X',end="\t")
			# print('\n')
		print('---------------------------------')

	def actionSpaceSample(self):
		return np.random.choice(self.possibleActions)


register(
    'gridworld-v0',
    entry_point='gridworld.gridworld:gridworld',
    #max_episode_steps=500,
)
