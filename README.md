# Q Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Design and implement a Python program that employs the Q-Learning algorithm to determine the optimal policy for a given Reinforcement Learning (RL) environment. Additionally, compare the state values obtained using Q-Learning with those obtained using the Monte Carlo method for the same RL environment.

## Q LEARNING ALGORITHM
1. Initialize Q-table and hyperparameters.

2. Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.

3. After training, derive the optimal policy from the Q-table.

4. Implement the Monte Carlo method to estimate state values.

5. Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION
```python
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action=lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(
        init_alpha,min_alpha,
        alpha_decay_ratio,
        n_episodes)
    epsilons=decay_schedule(
        init_epsilon,min_epsilon,
        epsilon_decay_ratio,
        n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:

### Optimal State value function
<img width="802" alt="image" src="https://github.com/Shavedha/q-learning/assets/93427376/03e94a47-3908-453f-a7f4-524860accaac">

### Optimal Action value function
<img width="658" alt="image" src="https://github.com/Shavedha/q-learning/assets/93427376/6d1e82a6-b508-4304-a55e-2a8c46a2fb8e">

### COMPARISON:
### Monte Carlo Method
<img width="885" alt="image" src="https://github.com/Shavedha/q-learning/assets/93427376/dd33191d-5eb6-4950-8beb-bb1b8944dbc0">

### Q - Learning Method
<img width="877" alt="image" src="https://github.com/Shavedha/q-learning/assets/93427376/ba6fb7bf-f6df-427e-90b0-ecf58f5d5457">


## RESULT:

Thus,a python program is developed to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.
