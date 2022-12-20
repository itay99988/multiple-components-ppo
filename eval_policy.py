"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py.
"""
import torch


def _log_summary(ep_len, ep_ret, ep_num):
    """
        Print to stdout what we've logged so far in the most recent episode.
        Parameters:
            None
        Return:
            None
    """
    # Round decimal places for more aesthetic logging messages
    failure_rate = str(round((ep_len - ep_ret)*(50 / ep_len), 2))
    ep_len = str(round(ep_len, 2))
    ep_ret = str(round(ep_ret, 2))

    # Print logging statements
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print("Failure Rate: {}%".format(failure_rate))
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)


def rollout(policy1, policy2, dual_if):
    """
        Returns a generator to roll out each episode given a trained policy and
        environment to test on.
        Parameters:
            policy - The trained policy to test
            env - The environment to evaluate the policy on

        Return:
            A generator object rollout, or iterable, which will return the latest
            episodic length and return on each iteration of the generator.
    """
    # Rollout until user kills process
    for _ in range(10):
        dual_if.reset()
        state1 = dual_if.get_compound_state(0)
        state2 = dual_if.get_compound_state(1)
        hidden_state1 = policy1.init_hidden()
        hidden_state2 = policy2.init_hidden()

        # number of timesteps so far
        t = 0

        # Logging data
        ep_len = 0  # episodic length
        ep_ret = 0  # episodic return

        while t < 200:
            t += 1

            # Query deterministic action from policy and run it
            state1 = torch.tensor(state1, dtype=torch.float).view(1, 1, -1)
            state2 = torch.tensor(state2, dtype=torch.float).view(1, 1, -1)

            # select the best action
            available_actions_idx1 = [dual_if.if1.get_transition_idx(tr_name) for tr_name in
                                      dual_if.if1.available_transitions()]
            available_actions_idx2 = [dual_if.if2.get_transition_idx(tr_name) for tr_name in
                                      dual_if.if2.available_transitions()]

            logits1, hidden_state1 = policy1(state1, hidden_state1)
            logits1 = logits1.view(-1)
            logits2, hidden_state2 = policy2(state2, hidden_state2)
            logits2 = logits2.view(-1)

            filtered_logits1 = logits1[available_actions_idx1]
            filtered_logits2 = logits2[available_actions_idx2]

            # choose based on probability
            # dist = Categorical(logits=filtered_logits)
            # action_idx = dist.sample()

            # choose action with maximal value
            action_idx1 = torch.argmax(filtered_logits1)
            action_idx2 = torch.argmax(filtered_logits2)

            real_action_idx1 = available_actions_idx1[action_idx1.numpy()]
            real_action_idx2 = available_actions_idx2[action_idx2.numpy()]

            state, reward = dual_if.step(real_action_idx1, real_action_idx2)

            # Sum all episodic rewards as we go along
            ep_ret += reward

        # Track episodic length
        ep_len = t

        # returns episodic length and return in this iteration
        yield ep_len, ep_ret


def eval_policy(policy1, policy2, dual_if):
    """
        The main function to evaluate our policy with. It will iterate a generator object
        "rollout", which will simulate each episode and return the most recent episode's
        length and return. We can then log it right after. And yes, eval_policy will run
        forever until you kill the process.
        Parameters:
            policy - The trained policy to test, basically another name for our actor model
            env - The environment to test the policy on
        Return:
            None
        NOTE: To learn more about generators, look at rollout's function description
    """
    # Rollout with the policy and environment, and log each episode's data
    for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy1, policy2, dual_if)):
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)
