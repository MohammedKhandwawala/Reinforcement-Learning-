from problem1 import *
from problem2 import *
from problem3 import *

iterations = 10000
number_of_arms = 1000

print("1 start")
Rt1,Ot1 = run_egreedy(e = 0.1,number_of_arms=number_of_arms,iterations = iterations)
print("2 start")
Rt2,Ot2 = run_softmax(temp = 0.1,number_of_arms=number_of_arms,iterations = iterations)
print("3 start")
Rt3,Ot3 = run_ucb(number_of_arms = number_of_arms,iterations = iterations)


plt.plot(Rt1)
plt.plot(Rt2)
plt.plot(Rt3)
plt.title("Comparison of Average performance of different action-value methods on the 10-armed testbed")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend(["e-greedy e = 0.1","Softmax T = 0.01","UCB1"])
plt.show()

plt.title("Comparison of Average performance of different action-value methods on the 10-armed testbed")
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.plot(Ot1)
plt.plot(Ot2)
plt.plot(Ot3)
plt.legend(["e-greedy e = 0.1","Softmax T = 0.01","UCB1"])
plt.show()
