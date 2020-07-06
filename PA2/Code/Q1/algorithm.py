import numpy as np
import matplotlib.pyplot as plt 
import gym 

import multiprocessing as mp


def maxAction(Q, state ,actions):
	values = np.array([Q[state,a] for a in actions])
	action = np.argmax(values)
	return actions[action]

def QLearning(numGames,env,alpha,gamma,eps):
	Q = {}
	steps = np.zeros(numGames)
	totalRewards  = np.zeros(numGames)
	for state in env.stateSpacePlus:
		for action in env.possibleActions:
			Q[state , action] = np.random.rand()
	# env.render()
	for i in range(numGames):
		if i%500 == 0:
			print("starting game",i)

		done = False
		epRewards = 0
		observation = env.reset()

		step_ct = 0
		while not done:
			rand = np.random.random()
			if rand < 1 - eps:
				action = maxAction(Q,observation, env.possibleActions)
			else :
				action = env.actionSpaceSample()

			observation_ , reward , done ,info = env.step(action)
			epRewards +=reward

			action_ = maxAction(Q,observation,env.possibleActions)
			Q[observation,action] = Q[observation,action] + alpha*(reward + gamma*Q[observation_,action_] - Q[observation,action])
			observation = observation_
			step_ct += 1

		if eps - 2/numGames > 0:
			eps -= 2/numGames
		else:
			eps = 0
		steps[i] = step_ct
		totalRewards[i] = epRewards

	return Q,steps,totalRewards

def SARSA(numGames,env,alpha,gamma,eps):
	Q = {}
	steps = np.zeros(numGames)
	totalRewards  = np.zeros(numGames)
	for state in env.stateSpacePlus:
		for action in env.possibleActions:
			Q[state , action] = np.random.rand()	
	# env.render()
	for i in range(numGames):
		if i%500 == 0:
			print("starting game",i)

		done = False	
		epRewards = 0
		observation = env.reset()

		rand = np.random.random()
		action = maxAction(Q,observation, env.possibleActions) if rand < (1-eps) else env.actionSpaceSample()
		step_ct = 0
		while not done:
			rand = np.random.random()
			observation_ , reward , done ,info = env.step(action)
			action_ = maxAction(Q,observation_, env.possibleActions) if rand < (1-eps) else env.actionSpaceSample()
			epRewards +=reward
			Q[observation,action] = Q[observation,action] + alpha*(reward + gamma*Q[observation_,action_] - Q[observation,action])
			action = action_
			observation = observation_
			step_ct+=1


		if eps - 2/numGames > 0:
			eps -= 2/numGames
		else:
			eps = 0
		steps[i] = step_ct
		totalRewards[i] = epRewards

	return Q,steps,totalRewards

def SARSA_lambda(numGames,env,alpha,gamma,eps,lbda):
	Q = {}

	steps = np.zeros(numGames)
	totalRewards  = np.zeros(numGames)

	MAX_STEPS = 1000	
	for state in env.stateSpacePlus:
		for action in env.possibleActions:
			Q[state , action] = np.random.rand()	
	print(lbda)
	# env.render()
	for i in range(numGames):
		if i%2 == 0:
			print("starting game",i)

		e = {}
		for state in env.stateSpacePlus:
			for action in env.possibleActions:
				e[state , action] = 0

		done = False
		epRewards = 0
		observation = env.reset()

		rand = np.random.random()
		action = maxAction(Q,observation, env.possibleActions) if rand < (1-eps) else env.actionSpaceSample()
		step_ct = 0
		for ii in range(MAX_STEPS):
			rand = np.random.random()
			observation_ , reward , done ,info = env.step(action)

			action_ = maxAction(Q,observation_, env.possibleActions) if rand < (1-eps) else env.actionSpaceSample()
			
			for keys in e:
				e[keys]*=gamma*lbda
			e[observation,action] += 1

			epRewards +=reward
			for keys in Q:
				Q[keys] = Q[keys] + alpha*lbda*e[keys]*(reward + gamma*Q[observation_,action_] - Q[observation,action])
			action = action_
			observation = observation_
			step_ct+=1

			if done:
				break


		if eps - 2/numGames > 0:
			eps -= 2/numGames
		else:
			eps = 0
		steps[i] = step_ct
		totalRewards[i] = epRewards

	return Q,steps,totalRewards

def getPolicy(Q):
	policy = np.zeros((12,12))
	i = 0
	j = 0
	mapping= {'N':0, 'E':1, 'W':2, 'S':3 }
	for state in env.stateSpacePlus:
		max_q = -99999999
		max_a = 1
		for action in env.possibleActions:
			if(Q[state,action] > max_q):
				max_a = action
				max_q = Q[state,action]
		policy[i][j] = mapping[max_a]
		j+=1
		if(j%12 == 0):
			i+=1
			j = 0
	return policy

def plotPolicy(goal, pol ):
	plt.rcParams['figure.figsize'] = [7,7]
	fig, ax = plt.subplots()
	ax.matshow(pol)
	for i in range(12):
		for j in range(12):
			if [j,i] == goal:
				print(1)
				ax.text(i,j,'G', va='center', ha='center')
			else:
				c = int(pol[j,i])
				arrow = {0:'↑', 1:'➜', 2:'←', 3:'↓' }
				ax.text(i, j, arrow[c], va='center', ha='center')		

###############################################################################################

if __name__ == '__main__':
	learning = 'SARSA_lambda'
	category = 3
	if category == 1:
		goalstate = 11
	elif category == 2:
		goalstate = 33
	else:	
		goalstate = 79
	
	from gridworld import gridworld
	env = gym.make('gridworld-v0')
	env.setInitParam(category,goalstate)
	alpha = 0.1
	gamma = 0.9
	eps = 0.1
	Na = 50
	numGames = 1000
	#lbda = [0,0.3,0.5,0.9,0.99,1.0]
	lbda = [0.9]
	Policies = []
	if learning == "SARSA_lambda":
		steps_avg = np.zeros((len(lbda),numGames))
		totalRewards_avg = np.zeros((len(lbda),numGames))
	else:
		steps_avg = np.zeros(numGames)
		totalRewards_avg = np.zeros(numGames)
	pol = np.zeros((12,12))

	if learning == "SARSA":
		for i in range(Na):
			Q,Steps,Rewards = SARSA(numGames,env,alpha,gamma,eps)
			Policies.append(getPolicy(Q))
			for j in range(numGames):
				steps_avg[j]+=Steps[j]/Na
				totalRewards_avg[j]+=Rewards[j]	/Na
	elif learning == "QLearning":
		for i in range(Na):
			Q,Steps,Rewards = SARSA(numGames,env,alpha,gamma,eps)
			Policies.append(getPolicy(Q))
			for j in range(numGames):
				steps_avg[j]+=Steps[j]/Na
				totalRewards_avg[j]+=Rewards[j]/Na
	else:
		for i in range(Na):
			for j in range(len(lbda)):
				Q,Steps,Rewards = SARSA_lambda(numGames,env,alpha,gamma,eps,lbda[j])
				Policies.append(getPolicy(Q))
				for k in range(numGames):
					steps_avg[j][k]+=Steps[k]/Na
					totalRewards_avg[j][k]+=Rewards[k]/Na	

	for i in range(12):
		for j in range(12):
			act = {0:0,1:0,2:0,3:0}
			for k in range(Na):
				act[int(Policies[k][i][j])]+=1
			pol[i][j] = max(act, key=act.get)

	if learning == 'SARSA_lambda':
		for i in range(len(lbda)):
			plt.plot(steps_avg[i])
		plt.legend([r'$\lambda = 0.9$',r'$\lambda = 0.3$',r'$\lambda = 0.5$',r'$\lambda = 0.9$',r'$\lambda = 0.99$',r'$\lambda = 1.0$'])
		plt.xlabel("Number of episodes")
		plt.ylabel("Average number of Steps")
		plt.show()

		for i in range(len(lbda)):
			plt.plot(totalRewards_avg[i])
		plt.legend([r'$\lambda = 0.9$',r'$\lambda = 0.3$',r'$\lambda = 0.5$',r'$\lambda = 0.9$',r'$\lambda = 0.99$',r'$\lambda = 1.0$'])
		plt.xlabel("Number of episodes")
		plt.ylabel("Average reward")
		plt.show()

		for i in range(1,len(lbda)):
			plt.plot(steps_avg[i])
		plt.legend([r'$\lambda = 0.9$',r'$\lambda = 0.5$',r'$\lambda = 0.9$',r'$\lambda = 0.99$',r'$\lambda = 1.0$'])
		plt.xlabel("Number of episodes")
		plt.ylabel("Average number of Steps")
		plt.show()

		for i in range(1,len(lbda)):
			plt.plot(totalRewards_avg[i])
		plt.legend([r'$\lambda = 0.9$',r'$\lambda = 0.5$',r'$\lambda = 0.9$',r'$\lambda = 0.99$',r'$\lambda = 1.0$'])
		plt.xlabel("Number of episodes")
		plt.ylabel("Average reward")
		plt.show()
	else:				
		plt.plot(steps_avg)
		plt.xlabel("Number of episodes")
		plt.ylabel("Average number of Steps")
		plt.show()

		plt.plot(totalRewards_avg)
		plt.xlabel("Number of episodes")
		plt.ylabel("Average reward")
		plt.show()
	goalstate = [goalstate//12,goalstate%12]
	print(goalstate)

	plotPolicy(goalstate,pol)
	plt.show()
