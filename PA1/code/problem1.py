import numpy as np
import matplotlib.pyplot as plt

def generate_action(mean = 0, sd = 1 , number_of_arms = 10):
	true_action = np.random.normal(mean,sd,number_of_arms)
	return true_action

def generate_test(action ,number_of_arms = 10):
	reward = np.random.normal(0,1,number_of_arms)+action;
	return reward

def e_greedy(true_action ,number_of_arms = 10,e = 0.1, iterations = 1000):
	optimal_arm = np.argmax(true_action)

	reward_t = np.zeros(iterations+1)
	optimal_t = np.zeros(iterations+1)

	Q = np.zeros(number_of_arms)
	N = np.zeros(number_of_arms)
	for i in range(1,iterations+1):
		reward = generate_test(true_action,number_of_arms = number_of_arms)
		if(np.random.random() > e):
			j = np.argmax(Q)
			N[j] += 1
			Q[j] = Q[j] + 1/(N[j])*(reward[j] - Q[j])
			reward_t[i] = reward[j]
			optimal_t[i] = N[optimal_arm]/i;
		else:
			j = np.random.randint(number_of_arms)
			N[j] += 1
			Q[j] = Q[j] + (1/N[j])*(reward[j] - Q[j])
			reward_t[i] = reward[j]
			optimal_t[i] = N[optimal_arm]/i;

	return reward_t,optimal_t


def run_egreedy(number_of_arms = 10,e = 0.1,iterations = 1000):
	true_reward = generate_action(number_of_arms = number_of_arms)
	Rt,Ot = e_greedy(true_reward,number_of_arms=number_of_arms,e=e,iterations =iterations)

	for j in range(1,2001):
		print(j)
		true_reward = generate_action(number_of_arms = number_of_arms)
		rt,ot = e_greedy(true_reward,number_of_arms = number_of_arms,e=e,iterations =iterations)
		Rt = Rt + rt
		Ot = Ot + ot
	return Rt/2000, Ot/2000

# Rt1,Ot1 = run_egreedy(0.1)
# Rt2,Ot2 = run_egreedy(0.01)
# Rt3,Ot3 = run_egreedy(0.05)
# Rt4,Ot4 = run_egreedy(0)


# plt.plot(Rt1)
# plt.plot(Rt2)
# plt.plot(Rt3)
# plt.plot(Rt4)
# plt.title("Average performance of e-greedy action-value methods on the 10-armed testbed")
# plt.xlabel("Steps")
# plt.ylabel("Average Reward")
# plt.legend(["e = 0.1","e = 0.01","e = 0.05","e = 0"])
# plt.show()

# plt.title("Average performance of e-greedy action-value methods on the 10-armed testbed")
# plt.xlabel("Steps")
# plt.ylabel("% Optimal Action")
# plt.plot(Ot1)
# plt.plot(Ot2)
# plt.plot(Ot3)
# plt.plot(Ot4)
# plt.legend(["e = 0.1","e = 0.01","e = 0.05","e = 0"])
# plt.show()