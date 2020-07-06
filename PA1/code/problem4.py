import numpy as np
import matplotlib.pyplot as plt

def generate_action(mean = 0, sd = 1 , number_of_arms = 10 , number_of_bandit = 2000):
	true_action = np.random.normal(mean,sd,(number_of_bandit,number_of_arms))
	return true_action

n_bandit = 2000 
k = 10 

trueReward = generate_action(number_of_arms = k)

trueRewardArm = np.argmax(trueReward,1) 
trueRewardMax = np.max(trueReward,1)
trueRewardMax = np.reshape(trueRewardMax,(n_bandit,1))

#eps_delta_pairs = [[0.1,0.1,'g'],[0.6,0.4,'k'],[0.4,0.6,'r'],[0.5,0.5,'b'],[0.7,0.3,'y']] 
eps_delta_pairs = [[0.9,0.9,'g'],[0.1,0.1,'k'],[0.4,0.6,'r'],[0.5,0.5,'b'],[0.7,0.3,'y']]


fig1 = plt.figure().add_subplot(111)
fig2 = plt.figure().add_subplot(111)

for eps_delta in eps_delta_pairs : 

	e = eps_delta[0] 
	delta = eps_delta[1] 

	print(e,delta)

	l = 1
	eps_l = e/4
	delta_l = delta/2
	trueReward_l = trueReward 
	k_l = k 
	trueRewardMax_l = trueRewardMax 
	rewards = []
	x_graph = 0
	opt_arms = []

	while( k_l > 1) : 

		opt_arms_l = 0
		sample_l = np.log(3.0/delta_l)*4/(eps_l**2)

		temp_R_l = np.zeros((n_bandit,int(k_l)))
	
		for cnt in range(int(sample_l)) : 
			temp_R = np.random.normal(trueReward_l,1)
			temp_R_l = temp_R_l+temp_R
			rewards.append(np.mean(temp_R))
			x_graph = x_graph+1

		avg_temp_R_l = temp_R_l/int(sample_l)
		medians_l = np.median(avg_temp_R_l,1) 

		trueReward_l_new = np.zeros((n_bandit,int(np.ceil(k_l/2))))

		print(trueReward_l.shape)

		for i in range(n_bandit) : 
			j1 = 0
			for j in range(int(k_l)) : 
				if avg_temp_R_l[i][j] >= medians_l[i] : 
					trueReward_l_new[i][j1] = trueReward_l[i][j]
					j1 = j1+1
					if j == trueRewardMax_l[i] : 
						opt_arms_l=opt_arms_l+1

		opt_arms.append(opt_arms_l*100/float(n_bandit))
		trueReward_l = trueReward_l_new
		trueRewardMax_l = np.argmax(trueReward_l,1)
		k_l *= 0.5 
		k_l = int(np.ceil(k_l))
		eps_l *= 0.75
		delta_l *= 0.5
		l+=1

	sample_l = np.log(3.0/delta_l)*4/(eps_l**2)
	temp_R_l = np.zeros((n_bandit,int(k_l)))
	for cnt in range(int(sample_l)) : 
		temp_R = np.random.normal(trueReward_l,1)
		temp_R_l = temp_R_l+temp_R
		rewards.append(np.mean(temp_R))
		x_graph = x_graph+1
	


	diff = abs(trueReward-trueReward_l)
	#cnt1 = n_bandit-np.count_nonzero(diff)
	#cnt2 = np.count_nonzero(diff<e)-cnt1
	#cnt3 = np.count_nonzero(diff>e)
	
	print( 'Number of Steps ',l)
	fig1.plot(range(x_graph),rewards,eps_delta[2])
	fig2.plot(range(1,l),opt_arms,eps_delta[2])

plt.rc('text',usetex=True)
fig1.title.set_text('MEA Average Reward versus Steps')
fig1.set_ylabel('Average Reward')
fig1.set_xlabel('Steps')
fig1.legend((r'$\epsilon$='+str(eps_delta_pairs[0][0])+r', $\delta$='+str(eps_delta_pairs[0][1]),r'$\epsilon$='+str(eps_delta_pairs[1][0])+r', $\delta$='+str(eps_delta_pairs[1][1]),r'$\epsilon$='+str(eps_delta_pairs[2][0])+r', $\delta$='+str(eps_delta_pairs[2][1]),r'$\epsilon$='+str(eps_delta_pairs[3][0])+r', $\delta$='+str(eps_delta_pairs[3][1]),r'$\epsilon$='+str(eps_delta_pairs[4][0])+r', $\delta$='+str(eps_delta_pairs[4][1])),loc='best')
fig2.title.set_text(r'MEA $\%$ Optimal Action Vs Steps')
fig2.set_xlabel('Steps')
fig2.set_ylabel(r'$\%$ Optimal Action')
fig2.set_ylim(0,110)
fig1.legend((r'$\epsilon$='+str(eps_delta_pairs[0][0])+r', $\delta$='+str(eps_delta_pairs[0][1]),r'$\epsilon$='+str(eps_delta_pairs[1][0])+r', $\delta$='+str(eps_delta_pairs[1][1]),r'$\epsilon$='+str(eps_delta_pairs[2][0])+r', $\delta$='+str(eps_delta_pairs[2][1]),r'$\epsilon$='+str(eps_delta_pairs[3][0])+r', $\delta$='+str(eps_delta_pairs[3][1]),r'$\epsilon$='+str(eps_delta_pairs[4][0])+r', $\delta$='+str(eps_delta_pairs[4][1])),loc='best')
plt.show()

