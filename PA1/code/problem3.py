import numpy as np
import matplotlib.pyplot as plt

def generate_action(mean = 0, sd = 1 , number_of_arms = 10):
	true_action = np.random.normal(mean,sd,number_of_arms)
	return true_action

def generate_test(action ,number_of_arms = 10 ):
	reward = np.random.normal(0,1,number_of_arms)+action;
	return reward

def UCB1(true_action ,number_of_arms = 10, iterations = 1000):
	optimal_arm = np.argmax(true_action)

	reward_t = np.zeros(iterations+1) 
	optimal_t = np.zeros(iterations+1)

	Q = generate_test(true_action,number_of_arms = number_of_arms)
	N = np.ones(number_of_arms)

	for i in range(1,iterations+1):
		reward = generate_test(true_action,number_of_arms=number_of_arms)
		Rt = Q + np.sqrt(2*np.log(i)/N)
		j = np.argmax(Rt)
		N[j] += 1
		Q[j] = Q[j] + 1/(N[j])*(reward[j] - Q[j])
		reward_t[i] = reward[j]
		optimal_t[i] = N[optimal_arm]/(i+number_of_arms);

	return reward_t,optimal_t


def run_ucb(number_of_arms = 10,iterations = 1000):
	true_reward = generate_action(number_of_arms = number_of_arms)
	Rt,Ot = UCB1(true_reward,number_of_arms = number_of_arms , iterations = iterations)

	for j in range(2000):
		print(j)
		true_reward = generate_action(number_of_arms = number_of_arms)
		rt,ot = UCB1(true_reward,number_of_arms = number_of_arms , iterations = iterations)
		Rt = Rt + rt
		Ot = Ot + ot
	return Rt/2000, Ot/2000

# Rt1,Ot1 = run_ucb()

# plt.plot(Rt1)
# plt.show()

# print(Ot1[:10])

# plt.plot(Ot1)
# plt.show()