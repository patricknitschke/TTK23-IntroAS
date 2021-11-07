from matplotlib import pyplot as plt
from numpy.core.fromnumeric import argmax
from gridWorld import gridWorld
from pendulum import pendulum
import numpy as np
import random

def show_action_value_function(env, Q):
    pos = {"U": (-0.15, -0.3), "D": (-0.15, 0.4), "L": (-0.45, 0.1), "R": (0.05, 0.1)}
    fig = env.render(show_state = False, show_reward = False)            
    for k in env.states():
        s = env.legal_states[k]
        for i, a in enumerate(env.actions(k)):
            fig.axes[0].annotate("{0:.2f}".format(Q[k, i]), (s[1] + pos[a][0], s[0] + pos[a][1]), size = 40/env.board_mask.shape[0], color = "r" if Q[k, i] == max(Q[k, :]) else "k")
    plt.show()
    
def show_policy(env, Q):
    fig = env.render(show_state = False, show_reward = False)
    action_map = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    for k in env.states():
        s = k if isinstance(k, tuple) else env.legal_states[k]
        if not env.terminal(s):
            fig.axes[0].annotate(action_map[env.actions(s)[np.argmax(Q[k, :])]], (s[1] - 0.1, s[0] + 0.1), size = 100/env.board_mask.shape[0])
    plt.show()


####################  Problem 1: Q-Learning #################### 
def Q_Learning(env, gamma, Q, alpha, epsilon):
    # Reset environment
    s, r, done = env.reset()

    """
    YOUR CODE HERE:
    Problem 1a) Implement Q-Learning
    
    Input arguments:
        - env     Is the environment
        - gamma   Is the discount rate
        - Q       Is the Q table
        - alpha   Is the learning rate
        - epsilon Is the probability of choosing greedy action
    
    Some usefull functions of the grid world environment
        - s_next, r, done = env.step(a)  Take action a and observe the next state, reward and environment termination
        - actions = env.actions()        List available actions in current state (is empty if state is terminal)
    """

    while not done:
        a = epsilon_greedy(env, s, Q, epsilon)
        action = env.actions()[a]
          
        s_, r, done = env.step(action)
        print(f"Moved to: {s_}, Reward: {r}, Done: {0}")   
        
        Q_a_max = Q[s_, argmax(Q[s_,:])]
        print(f"Next state max Q = {Q_a_max}")

        q = Q[s,a]
        Q[s,a] = Q[s,a] + alpha*(r + gamma*(Q_a_max) - Q[s,a])
        print(f"Update = {Q[s,a] - q}")

        s = s_

    return Q

# Epsilon greedy policy
def epsilon_greedy(env, s, Q, epsilon=0.1):
    a_optimal = argmax(Q[s,:])
    A = list(range(len(env.actions())))
    return random.choices([a_optimal] + A, [1-epsilon] + [epsilon/len(A) for _ in range(len(A))])[0]


def Q_Learning_cont(env, gamma, Q, alpha, epsilon):
    # Reset environment
    s, r, done = env.reset()
    s[0] = round(s[0]*10)
    s = [int(i) for i in s]

    while not done:
        a = epsilon_greedy_cont(env, s, Q, epsilon)
        action = env.actions()[a]
          
        s_, r, done = env.step(action)
        s_[0] = int(s_[0]*10)
        s_ = [int(i) for i in s_]

        print(f"Moved to: {s_}, Reward: {r}, Done: {0}")   
        
        Q_a_max = Q[s_[0], s_[1], argmax(Q[s_[0],s_[1], :])]
        print(f"Next state max Q = {Q_a_max}")

        q = Q[s[0],s[1],a]
        Q[s[0],s[1],a] = Q[s[0],s[1],a] + alpha*(r + gamma*(Q_a_max) - Q[s[0],s[1],a])
        print(f"Update = {Q[s[0],s[1],a] - q}")

        s = s_

    return Q

# Epsilon greedy policy
def epsilon_greedy_cont(env, s, Q, epsilon=0.1):
    a_optimal = argmax(Q[s[0],s[1], :])
    A = list(range(len(env.actions())))
    return random.choices([a_optimal] + A, [1-epsilon] + [epsilon/len(A) for _ in range(len(A))])[0]

####################  Problem 2: SARSA #################### 
def SARSA(env, gamma, Q, alpha, epsilon):
    # Reset environment
    s, r, done = env.reset()

    """
    YOUR CODE HERE:
    Problem 2a) Implement SARSA
    
    Input arguments:
        - env     Is the environment
        - gamma   Is the discount rate
        - Q       Is the Q table
        - alpha   Is the learning rate
        - epsilon Is the probability of choosing greedy action
    
    Some usefull functions of the grid world environment
        - s_next, r, done = env.step(a)  Take action a and observe the next state, reward and environment termination
        - actions = env.actions()        List available actions in current state (is empty if state is terminal)
    """
    
    a = epsilon_greedy(env, s, Q, epsilon)

    while not done:
        action = env.actions()[a]
        s_, r, done = env.step(action)
        a_ = epsilon_greedy(env, s_, Q, epsilon)
        
        Q[s,a] = Q[s,a] + alpha*(r + gamma*Q[s_,a_] - Q[s,a])
        
        s = s_
        a = a_

    return Q

def SARSA_cont(env, gamma, Q, alpha, epsilon):
    # Reset environment
    s, r, done = env.reset()
    s[0] = s[0]*10
    s = [int(i) for i in s]

    """
    YOUR CODE HERE:
    Problem 2a) Implement SARSA
    
    Input arguments:
        - env     Is the environment
        - gamma   Is the discount rate
        - Q       Is the Q table
        - alpha   Is the learning rate
        - epsilon Is the probability of choosing greedy action
    
    Some usefull functions of the grid world environment
        - s_next, r, done = env.step(a)  Take action a and observe the next state, reward and environment termination
        - actions = env.actions()        List available actions in current state (is empty if state is terminal)
    """
    
    a = epsilon_greedy_cont(env, s, Q, epsilon)

    while not done:
        action = env.actions()[a]
        s_, r, done = env.step(action)
        s_[0] = s_[0]*10
        s_ = [int(i) for i in s_]
        a_ = epsilon_greedy_cont(env, s_, Q, epsilon)
        
        update = (r + gamma*Q[s_[0], s[1], a_] - Q[s[0], s[1], a])
        Q[s[0], s[1], a] = Q[s[0], s[1], a] + alpha*update
        print(f"Update: {update}")
        
        s = s_
        a = a_

    return Q

if __name__ == "__main__":
    """
    Note that this code has been written for python 3.x, and requiers the numpy, matplotlib
    and scipy packages.
    """

    # Import the environment from file
    filename = "gridworlds/tiny.json"
    env = gridWorld(filename)

    # Render image
    fig = env.render(show_state = True)
    plt.show()


    """
    Problem 1 (Run Q-Learning)
    
    Below is the code for running Q-Learning, feel free to change the code, and tweek the parameters.
    """
    gamma = 0.99     # Discount rate
    alpha = 0.1     # Learning rate
    epsilon = 0.2   # Probability of exploring
    episodes = 5000 # Number of episodes

    Q = np.zeros([len(env.states()), 4])
    for i in range(episodes):
        Q_Learning(env, gamma, Q, alpha, epsilon)

    # Render Q-values and policy 
    show_action_value_function(env, Q)
    show_policy(env, Q)


    """
    Problem 2 (Run SARSA)
    
    Below is the code for running SARSA, feel free to change the code, and tweek the parameters.
    """
    gamma = 0.99    # Discount rate
    alpha = 0.1     # Learning rate
    epsilon = 0.2   # Probability of exploration
    episodes = 5000 # Number of episodes

    Q = np.zeros([len(env.states()), 4])
    for i in range(episodes):
        SARSA(env, gamma, Q, alpha, epsilon)

    # Render Q-values and policy 
    show_action_value_function(env, Q)
    show_policy(env, Q)


    """
    Problem 3) Pendulum 
    
    For this problem the aim is to use either Q-Learning or SARSA for the pendulum environment. You should be able to
    reuse most of the Q-Learning and SARSA algorithms form the preveous problems, but you may have to make two main 
    changes:
        
        1)  The states of the pendulum environment are continous, you must create a discretization of the state so 
            that it can be stored in a Q-table. 
    
        2)  The state of the pendulum has multiple elements (s = [theta, theta_dot]) you must change the code so that
            Q-table is able to store this. The simplest way ofdoing this is to create a three dimensional Q-Table in 
            the following way:  Q[theta, theta_dot, action]
    """

    # Create instance of pendulum environment
    env = pendulum()
    fig = env.render()

    gamma   = 0.99  # Discount rate
    alpha   = 0.2   # Learning rate
    epsilon = 0.2   # Probability of exploring
    episodes = 1000 # Number of episodes

    # raise Exception("Problem 3a) Choose your Q-Table to fit discretization")
    S = [theta for theta in range(-31, 31)]
    S2 = [theta_dot for theta_dot in range(-10, 10)]

    Q = np.zeros([len(S), len(S2), 3])
    
    for i in range(episodes):
        #raise Exception("Problem 3a) Choose either Q-Learning or SARSA, and modify it to work with the pendulum environment")
        Q_Learning_cont(env, gamma, Q, alpha, epsilon)

    # Plot the value function
    V = np.max(Q, axis=2)   # Value function is given as V(s) = max_a Q(s, a) 
    V[V==0] = np.nan        # Assume states with zeros are not visited and set them to NAN (gives better plot)

    plt.figure()
    plt.imshow(V, cmap="viridis")
    plt.colorbar()
    plt.show()


    # Run greedy policy for 10 second to see how the trained policy behaves
    plt.ion()
    fig = env.render()
    s, _, _ = env.reset([np.pi, 0])
    for i in range(200):
        #raise Exception("Problem 3a) Change code below to use you discretization. If you do, you should get a cool animation")
        s[0] = s[0]*10
        s = [int(i) for i in s]
        a = np.argmax(Q[s[0], s[1], :])
        s, _, _ = env.step(a)
        plt.pause(env.step_size)