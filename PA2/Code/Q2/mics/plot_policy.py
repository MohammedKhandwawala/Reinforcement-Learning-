import matplotlib.pyplot as plt 
import numpy as np 

#theta = np.array([[-1,0,0],[0,-1,0]])
theta = np.array([[-2.00862838,  0.0105143 ,  0.0095315 ],[ 0.00980043, -2.02697878 ,-0.00562391]])

x = np.linspace(-1,1,30)
y = np.linspace(-1,1,30)

mesh1 = np.zeros((30,30))
mesh2 = np.zeros((30,30))
for i in range(30):
	for j in range(30):
		s = np.array([x[i],y[j],1])
		vect = theta.dot(s)
		mesh1[i][j] = vect[0]
		mesh2[i][j] = vect[1]

fig, ax = plt.subplots()
q = ax.quiver(x, y, mesh2,mesh1)
ax.quiverkey(q, X=0.3, Y=1.1, U=10,label="",
              labelpos='E')
ax.set_title("Policy Learned for Visham for "+r'$\gamma$' + " = 0.9")
plt.show()

