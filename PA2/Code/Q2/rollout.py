#!/usr/bin/env python

import click
import numpy as np
import gym
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def include_bias(x):
    return np.append(x,1.0)

def chakra_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=1)		 

# @click.command()
# @click.argument("env_id", type=str, default="chakra")

def trajectory(env,theta,rng,rendr):
    states = []
    actions = []
    reward = []
    ob = env.reset()

    done = False

    while not done:
        action = chakra_get_action(theta, ob, rng=rng)
        next_ob, rw, done, _ = env.step(action)
        states.append(ob)
        actions.append(action)

        ob = next_ob
        if rendr :
            env.render(mode='human')
        reward.append(rw)
        if len(reward) >= 500:
            break
    return states , actions ,reward

def compute_gradient(state,action,theta):
    state = include_bias(state)
    mean = theta.dot(state)
    vect = action - mean
    # vect = vect.reshape(vect.shape[0],1)
    # state = state.reshape(state.shape[0],1)
    grad = np.outer(vect, state)
    return grad



def main(env_id):
    print(env_id)
    # Register the environment
    rng = np.random.RandomState(13)
    if env_id == 'chakra':
        from rlpa2 import chakra
        env = gym.make('chakra-v0')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    elif env_id == 'visham':
        from rlpa2 import visham
        env = gym.make('visham-v0')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' ")

    env.seed(13)
    # Initialize parameters
    theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))
    print(theta)
    ret_avg = np.zeros(int(episodes/batch_size))
    v_avg = np.zeros((103,103))
    lr = alpha

    for i in range(int(episodes/batch_size)):
        v = np.zeros((103,103))
        theta_d = np.zeros(theta.shape)
        reward_avg = 0
        retrn_avg = 0
        for j in range(batch_size):
            if j == batch_size-1 and i%50 == 0 and i!=0:
                states,action,reward = trajectory(env,theta,rng,True)
            else:
                states,action,reward = trajectory(env,theta,rng,False)
            
            ret_avg[i]+=len(states)
            retrn = np.zeros(len(reward))
            avg_reward = 0
            dis_return = 0

            for l in range(len(reward) - 1):
                x1,y1 = states[l]
                x2,y2 = states[l+1]
                # print(int(50*(x2+1.025)),int(50*(y2+1.025)),x2,y2,states[l],l)
                v[int(50*(x1+1.025))][int(50*(y1+1.025))]+=alpha_v*(-np.linalg.norm(states[l]) + gamma*v[int(50*(x2+1.025))][int(50*(y2+1.025))] - v[int(50*(x1+1.025))][int(50*(y1+1.025))]);

            for l in range(len(reward)-1,-1,-1):
                dis_return = gamma * dis_return + reward[l]
                retrn[l] = dis_return
                avg_reward+=reward[l]/len(reward)

            retrn_avg += sum(retrn)
            retrn_std = np.std(retrn)
            gradient_pi = np.zeros(theta.shape)

            for l in range(len(states)-1,-1,-1):
                dist = (np.linalg.norm(states[l]))
                advantage = retrn[l] - sum(retrn)/len(retrn)
                gradient_pi += advantage*compute_gradient(states[l],action[l],theta)
                # gradient_pi = gradient_pi/(np.linalg.norm(gradient_pi)+1e-8)
            theta_d += gradient_pi
        
        theta_d = theta_d/(np.linalg.norm(theta_d)+1e-8)
        theta+=lr*theta_d
        v_avg += v
        # lr = alpha - i*(10**(-2) - 10**(-6))*batch_size/episodes
        # theta = theta/(np.linalg.norm(theta)+1e-8)
        print("Average for",i," ",ret_avg[i]/(batch_size),retrn_avg/(batch_size))
        print(theta)
    return theta,ret_avg/batch_size,v_avg
        

        
if __name__ == "__main__":
    if(len(sys.argv) == 7):
        env = sys.argv[1]
        episodes = sys.argv[2]
        batch_size = sys.argv[3]
        alpha = sys.argv[4]
        gamma = sys.argv[5]
        render = sys.argv[6]
    else:
        env = "visham"
        episodes = 120000
        batch_size = 100
        alpha = 1e-2
        alpha_v = 0.1
        gamma = 0.01  
        render = False
    
    theta,avgRet,v_func = main(env)
    print(theta)
    plt.plot(avgRet)
    plt.show()
    x = np.linspace(-1,1,103)
    y = np.linspace(-1,1,103)
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, v_func/(episodes), cmap=cm.coolwarm) 
    ax.set_title("Value Function Plot")
    plt.show()

