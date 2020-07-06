import numpy as np
import matplotlib.pyplot as plt

def generate_action(mean = 0, sd = 1 , number_of_arms = 10):
	true_action = np.random.normal(mean,sd,number_of_arms)
	return true_action

def generate_test(action ,number_of_arms = 10 ):
	reward = np.random.normal(0,1,number_of_arms)+action;
	return reward

def softmax(true_action,temp = 1 ,number_of_arms = 10, iterations = 1000):

	optimal_arm = np.argmax(true_action)

	reward_t = np.zeros(iterations)
	optimal_t = np.zeros(iterations)

	Q = np.zeros(number_of_arms)  
	N = np.zeros(number_of_arms) 

	for i in range(iterations):
		reward = generate_test(true_action,number_of_arms = number_of_arms)
		softmax = np.exp(Q/temp)/sum(np.exp(Q/temp))
		j = np.random.choice(a=range(number_of_arms),p = softmax)
		N[j] += 1
		Q[j] = Q[j] + (1/N[j])*(reward[j] - Q[j])
		reward_t[i] = reward[j]
		optimal_t[i] = N[optimal_arm]/(i+1);	
	return reward_t,optimal_t


def run_softmax(temp = 1,number_of_arms = 10,iterations = 1000):
	true_reward = generate_action(number_of_arms =number_of_arms)
	Rt,Ot = softmax(true_reward,temp=temp,number_of_arms = number_of_arms,iterations = iterations)

	for j in range(2000):
		print(j)
		true_reward = generate_action(number_of_arms = number_of_arms)
		rt,ot = softmax(true_reward,temp=temp,number_of_arms = number_of_arms,iterations = iterations)
		Rt = Rt + rt
		Ot = Ot + ot
	return Rt/2000, Ot/2000

# Rt1,Ot1 = run_softmax(0.01)
# Rt2,Ot2 = run_softmax(0.1)
# Rt3,Ot3 = run_softmax(1)
# Rt4,Ot4 = run_softmax(1.5)

# plt.plot(Rt1)
# plt.plot(Rt2)
# plt.plot(Rt3)
# plt.plot(Rt4)
# plt.title("Average performance of softmax action selection method on the 10-armed testbed")
# plt.xlabel("Steps")
# plt.ylabel("Average Reward")
# plt.legend(["T = 0.01","T = 0.1","T = 1","T = 1.5"])
# plt.show()

# plt.plot(Ot1)
# plt.plot(Ot2)
# plt.plot(Ot3)
# plt.plot(Ot4)
# plt.title("Average performance of softmax action selection method on the 10-armed testbed")
# plt.xlabel("Steps")
# plt.ylabel("Average Reward")
# plt.legend(["T = 0.01","T = 0.1","T = 1","T = 1.5"])
# plt.show()