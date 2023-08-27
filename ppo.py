import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from analyze_dp_results import analyze_joint_policy


class PPO:

    def __init__(self, policy_class, mult_if, **hyperparameters):
        """
			Initializes the PPO model, including hyperparameters.
			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
			Returns:
				None
		"""
        # Extract number of interfaces
        self.if_count = mult_if.if_count
        # Initialize lists
        self.tr_dim, self.input_dim, self.actor, self.critic, self.actor_optim, self.critic_optim = [], [], [], [], [], []

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract multiple interface environment information
        self.mult_if = mult_if

        for i in range(self.if_count):
            curr_tr_dim = len(mult_if.get_if(i).transitions)
            self.tr_dim.append(curr_tr_dim)
            curr_input_dim = len(mult_if.get_if(i).states) * curr_tr_dim * self.history_len
            self.input_dim.append(curr_input_dim)

            # Initialize actor and critic networks for all interfaces
            self.actor.append(policy_class(self.input_dim[i], self.tr_dim[i]))  # ALG STEP 1
            self.critic.append(policy_class(self.input_dim[i], 1))

            # Initialize optimizers for actor and critic
            self.actor_optim.append(Adam(self.actor[i].parameters(), lr=self.lr))
            self.critic_optim.append(Adam(self.critic[i].parameters(), lr=self.lr))

        self.end_condition = self.end_cond_threshold * self.max_timesteps_per_episode

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [[] for _ in range(self.if_count)],  # episodic returns in batch
            'avg_ep_returns': [[] for _ in range(self.if_count)],  # Average episodic returns for each batch
            'actor_losses': [],  # losses of actor network in current iteration
        }

    def learn(self, total_timesteps):
        """
			Train the actor and critic networks. Here is where the main PPO algorithm resides.
			Parameters:
				total_timesteps - the total number of timesteps to train for
			Return:
				None
		"""
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far

        for i in range(self.if_count):
            self.mult_if.set_exploration_status(i, True)

        while t_so_far < total_timesteps:  # ALG STEP 2
            self.initial_temperature /= self.temperature_decay

            # Here we're collecting our batch simulations
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()  # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens[0])

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # convert batch_obs to rnn batch, for each one of the interfaces
            batch_size = int(self.timesteps_per_batch / self.max_timesteps_per_episode)

            V = [None] * self.if_count
            A_k = [None] * self.if_count

            for if_idx in range(self.if_count):
                batch_obs[if_idx] = batch_obs[if_idx].view(batch_size, self.max_timesteps_per_episode, -1)

                # Calculate advantage at k-th iteration
                V[if_idx], _ = self.evaluate(batch_obs[if_idx], batch_acts[if_idx], if_idx)
                A_k[if_idx] = batch_rtgs[if_idx] - V[if_idx].detach()  # ALG STEP 5

                # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
                # isn't theoretically necessary, but in practice it decreases the variance of
                # our advantages and makes convergence much more stable and faster. I added this because
                # solving some environments was too unstable without it.
                A_k[if_idx] = (A_k[if_idx] - A_k[if_idx].mean()) / (A_k[if_idx].std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                curr_log_probs = [None] * self.if_count
                ratios = [None] * self.if_count
                surr1, surr2 = [None] * self.if_count, [None] * self.if_count
                actor_loss, critic_loss = [None] * self.if_count, [None] * self.if_count

                for if_idx in range(self.if_count):
                    # Calculate V_phi and pi_theta(a_t | s_t)
                    V[if_idx], curr_log_probs[if_idx] = self.evaluate(batch_obs[if_idx], batch_acts[if_idx], if_idx)

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    # NOTE: we just subtract the logs, which is the same as
                    # dividing the values and then canceling the log with e^log.
                    # For why we use log probabilities instead of actual probabilities,
                    # here's a great explanation:
                    # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                    # TL;DR makes gradient ascent easier behind the scenes.
                    ratios[if_idx] = torch.exp(curr_log_probs[if_idx] - batch_log_probs[if_idx])

                    # Calculate surrogate losses.
                    surr1[if_idx] = ratios[if_idx] * A_k[if_idx]
                    surr2[if_idx] = torch.clamp(ratios[if_idx], 1 - self.clip, 1 + self.clip) * A_k[if_idx]

                    # Calculate actor and critic losses.
                    # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                    # the performance function, but Adam minimizes the loss. So minimizing the negative
                    # performance function maximizes it.
                    actor_loss[if_idx] = (-torch.min(surr1[if_idx], surr2[if_idx])).mean()
                    critic_loss[if_idx] = nn.MSELoss()(V[if_idx], batch_rtgs[if_idx])

                    # Calculate gradients and perform backward propagation for actor and critic network
                    self.actor_optim[if_idx].zero_grad()
                    actor_loss[if_idx].backward(retain_graph=True)
                    self.actor_optim[if_idx].step()
                    self.critic_optim[if_idx].zero_grad()
                    critic_loss[if_idx].backward()
                    self.critic_optim[if_idx].step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss[0].detach())

            # end condition
            for if_idx in range(self.if_count):
                last_ear = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews'][if_idx]])
                self.logger['avg_ep_returns'][if_idx].append(last_ear)

            avg_ep_return = np.mean(self.logger['avg_ep_returns'][0][-8:])
            
            if avg_ep_return >= self.end_condition:
                for if_idx in range(self.if_count):
                    torch.save(self.actor[if_idx].state_dict(), f'./ppo_actor{if_idx}.pth')
                    torch.save(self.critic[if_idx].state_dict(), f'./ppo_critic{if_idx}.pth')
                break

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                for if_idx in range(self.if_count):
                    torch.save(self.actor[if_idx].state_dict(), f'./ppo_actor{if_idx}.pth')
                    torch.save(self.critic[if_idx].state_dict(), f'./ppo_critic{if_idx}.pth')

    def rollout(self):
        """
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.
			Parameters:
				None
			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""

        # get exploration values for the interfaces
        exp_status = [self.mult_if.get_exploration_status(i) for i in range(self.if_count)]

        # states lists (for each interface)
        state = [[] for _ in range(self.if_count)]
        next_state = [[] for _ in range(self.if_count)]
        actor_hidden_state = [[] for _ in range(self.if_count)]

        # Batch data. For more details, check function header.
        batch_obs = [[] for _ in range(self.if_count)]
        batch_acts = [[] for _ in range(self.if_count)]
        batch_log_probs = [[] for _ in range(self.if_count)]
        batch_rews = [[] for _ in range(self.if_count)]
        batch_rtgs = [[] for _ in range(self.if_count)]
        batch_lens = [[] for _ in range(self.if_count)]

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = [[] for _ in range(self.if_count)]

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        if_suspend = [self.mult_if.get_if(i).get_suspend_lst(self.max_timesteps_per_episode)
                      for i in range(self.if_count)]

        # episode lengths, excluding suspended timesteps
        real_ep_lens = [self.max_timesteps_per_episode - len(if_suspend[i]) for i in range(self.if_count)]

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = [[] for _ in range(self.if_count)]  # rewards collected per episode

            # Reset the environment. Note that obs is short for observation.
            self.mult_if.reset()

            state = [self.mult_if.get_compound_state(i) for i in range(self.if_count)]
            actor_hidden_state = [self.actor[i].init_hidden() for i in range(self.if_count)]
            log_prob = [0] * self.if_count

            checked_interface = [False] * self.if_count
            reward = [0] * self.if_count
            action = [0] * self.if_count

            triggered_tr = [True] * self.if_count
            actions_count = [0] * self.if_count
            local_counter = [0] * self.if_count
            success_counter = [0] * self.if_count
            post_eat_counter = [0] * self.if_count
            timestep_count = 0

            t += self.max_timesteps_per_episode

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            while max(actions_count) < self.max_timesteps_per_episode:
                # t += 1  # Increment timesteps ran this batch so far

                for j in range(self.if_count):
                    if timestep_count in if_suspend[j]:
                        triggered_tr[j] = False
                        checked_interface[j] = True
                    else:
                        triggered_tr[j] = True

                # Calculate action and make a step, in all interfaces
                for i in range(self.if_count):
                    if triggered_tr[i]:
                        action[i], log_prob[i], actor_hidden_state[i], fil_logits = self.get_action(state[i],
                                                                                         actor_hidden_state[i],
                                                                                         if_idx=i,
                                                                                         exploration=exp_status[i])
                        actions_count[i] += 1

                # check if two different interfaces tried to communicate with each other
                for i in range(self.if_count):
                    for j in range(i + 1, self.if_count):
                        if self.mult_if.check_comm_attempt(i, j, action[i], action[j]) and triggered_tr[i] and triggered_tr[j]:
                            next_state[i], next_state[j], reward[i] = self.mult_if.step(i, j, action[i], action[j])
                            reward[j] = reward[i]
                            checked_interface[i] = checked_interface[j] = True

                            # check if post eat action
                            if self.mult_if.get_if(i).get_transition_by_idx(action[i]).source_state == 'g7':
                                post_eat_counter[i] += 1
                            if self.mult_if.get_if(j).get_transition_by_idx(action[j]).source_state == 'g7':
                                post_eat_counter[j] += 1

                # before handling the other actions, count successes for each interface.
                for i in range(self.if_count):
                    if reward[i] > 0:
                        success_counter[i] += 1

                # handle the other actions (missed communication or local actions)
                for i in range(self.if_count):
                    if not checked_interface[i]:
                        if self.mult_if.get_if(i).get_transition_by_idx(action[i]).is_global():
                            next_state[i], reward[i] = self.mult_if.missed_global_step(action[i], i)
                        else:
                            next_state[i], reward[i] = self.mult_if.local_step(action[i], i)
                            local_counter[i] += 1
                        checked_interface[i] = True

                # additional reward in the end of the episode
                if max(actions_count) == self.max_timesteps_per_episode:
                    # local_actions_ratio = [local_counter[i] / real_ep_lens[i] for i in range(self.if_count)]

                    # local action is eating -> every philosopher should eat
                    # uniform_eating_bonus = 10 * min(local_counter[::2])
                    # post eat
                    uniform_post_eat_bonus = 10 * min(post_eat_counter[::2])
                    for i in range(self.if_count):
                        reward[i] += uniform_post_eat_bonus

                # Track recent observation, reward, action, and action log probability (if there was a progress)
                for i in range(self.if_count):
                    # we want to save the reward of last timestep anyway.
                    if triggered_tr[i] or max(actions_count) == self.max_timesteps_per_episode:
                        batch_obs[i].append(state[i])
                        ep_rews[i].append(reward[i])
                        batch_acts[i].append(action[i])
                        batch_log_probs[i].append(log_prob[i])
                        state[i] = next_state[i]
                    # in this edge case, increase the actions counter by 1.
                    if not triggered_tr[i] and max(actions_count) == self.max_timesteps_per_episode:
                        actions_count[i] += 1

                # reset rewards and actions
                reward = [0] * self.if_count
                action = [0] * self.if_count
                checked_interface = [False] * self.if_count

                timestep_count += 1

            # Track episodic lengths and rewards
            for i in range(self.if_count):
                batch_lens[i].append(self.max_timesteps_per_episode)

            # Fill the shorter episodes with padding
            for i in range(self.if_count):
                if actions_count[i] < self.max_timesteps_per_episode:
                    pad_length = self.max_timesteps_per_episode - actions_count[i]
                    for _ in range(pad_length):
                        batch_obs[i].append([0]*len(state[i]))
                        ep_rews[i].append(0)
                        batch_acts[i].append(0)
                        batch_log_probs[i].append(torch.tensor(0, dtype=torch.float))

            # Do this anyway
            for i in range(self.if_count):
                batch_rews[i].append(ep_rews[i])

        # this is just for analyzing and printing the different policies (can be applied for every analysis)
        policies = []
        for i in range(self.if_count):
            policies.append(self.get_actions_seq(batch_acts[i][0:self.max_timesteps_per_episode], i))
            # print actions
            print(f"{self.get_actions_seq(batch_acts[i][0:self.max_timesteps_per_episode], i)} \n\n")

            # Reshape data as tensors in the shape specified in function description, before returning
            batch_obs[i] = torch.tensor(batch_obs[i], dtype=torch.float)
            batch_acts[i] = torch.tensor(batch_acts[i], dtype=torch.float)
            batch_log_probs[i] = torch.tensor(batch_log_probs[i], dtype=torch.float)
            batch_rtgs[i] = self.compute_rtgs(batch_rews[i])  # ALG STEP 4

            # Log the episodic returns and episodic lengths in this batch.
            self.logger['batch_rews'][i] = batch_rews[i]

        # print the analysis of the joint policy (who ate and when)
        analyze_joint_policy(policies, "eat1")

        self.logger['batch_lens'] = batch_lens[0]

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


    def compute_rtgs(self, batch_rews):
        """
			Compute the Reward-To-Go of each timestep in a batch given the rewards.
			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, state, actor_hidden_state, if_idx, exploration=True):
        """
			Queries an action from the actor network, should be called from rollout.
			Parameters:
				state - the observation at the current timestep
			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""

        temperature = max(1, self.initial_temperature)

        available_actions_idx = [self.mult_if.get_if(if_idx).get_transition_idx(tr_name, tr_target_if=i)
                                 for tr_name, i in self.mult_if.get_if(if_idx).available_transitions_with_if()]

        # Query the actor network
        state = torch.tensor(state, dtype=torch.float).view(1, 1, -1)
        logits, actor_hidden_state = self.actor[if_idx](state, actor_hidden_state)
        logits = logits.view(-1)
        filtered_logits = logits[available_actions_idx] / temperature
        dist = Categorical(logits=filtered_logits)
        if exploration:
            # Sample an action from the distribution
            action = dist.sample()
        else:
            # Select the action with the highest probability
            action = torch.argmax(filtered_logits)

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return available_actions_idx[action.detach().numpy()], log_prob.detach(), actor_hidden_state, filtered_logits

    def evaluate(self, batch_obs, batch_acts, if_idx):
        """
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.
			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)
			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""

        batch_size = int(self.timesteps_per_batch / self.max_timesteps_per_episode)
        actor_hidden_state = self.actor[if_idx].init_hidden(batch_size=batch_size)
        critic_hidden_state = self.critic[if_idx].init_hidden(batch_size=batch_size)

        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V, critic_hidden_state = self.critic[if_idx](batch_obs, critic_hidden_state)
        V = V.view(-1)

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        logits, actor_hidden_state = self.actor[if_idx](batch_obs, actor_hidden_state)
        logits = logits.view(-1, self.tr_dim[if_idx])
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def get_actions_seq(self, act_lst, if_num):
        return [(self.mult_if.get_if(if_num).get_transition_by_idx(act).name,
                self.mult_if.get_if(if_num).get_transition_by_idx(act).target_if_idx) for act in act_lst]

    def _init_hyperparameters(self, hyperparameters):
        """
			Initialize default and custom values for hyperparameters
			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.
			Return:
				None
		"""
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.history_len = 1  # History length
        self.timesteps_per_batch = 500  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 50  # Max number of timesteps per episode
        self.n_updates_per_iteration = 5  # Number of times to update actor/critic per iteration
        self.lr = 0.005  # Learning rate of actor optimizer
        self.gamma = 0.95  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.initial_temperature = 2
        self.temperature_decay = 1.008
        self.end_cond_threshold = 0.99

        # Miscellaneous parameters
        self.save_freq = 10  # How often we save in number of iterations
        self.seed = None  # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert (type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
			Print to stdout what we've logged so far in the most recent batch.
			Parameters:
				None
			Return:
				None
		"""
        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        avg_ep_rews = []
        for if_idx in range(self.if_count):
            avg_ep_rews.append(np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews'][if_idx]]))
        
        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        for if_idx in range(self.if_count):
            print(f"Interface {if_idx} Average Episodic Return: {str(round(avg_ep_rews[if_idx], 2))}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # save rewards figure
        self.make_rewards_graph(f'Average Rewards as a function of Learning Epochs',
                                f'Learning Epochs',
                                f'Average Rewards',
                                [self.mult_if.get_if(i).name for i in range(self.if_count)]
                                )

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = [[] for _ in range(self.if_count)]
        self.logger['actor_losses'] = []

    def make_rewards_graph(self, title_label, xlabel, ylabel, ifs_labels):
        path = f"./figures/rew_graphs.png"
        
        avg_ep_rews = []
        for i in range(self.if_count):
            avg_ep_rews.append([np.sum(ep_rews) for ep_rews in self.logger['avg_ep_returns'][i]])
        
        batch_count = len(avg_ep_rews[0])
        x_axis = list(range(1, batch_count+1))

        for i in range(self.if_count):
            plt.plot(x_axis, avg_ep_rews[i], label=ifs_labels[i])

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title_label)
        plt.legend()
        plt.savefig(path)
        plt.clf()
