"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py.
"""
import torch


def _log_summary(ep_len, ep_ret_lst, ep_num):
    """
        Print to stdout what we've logged so far in the most recent episode.
        Parameters:
            None
        Return:
            None
    """
    if_count = len(ep_ret_lst)
    
    # Round decimal places for more aesthetic logging messages
    ep_len = str(round(ep_len, 2))

    ep_ret_str = []
    for i in range(if_count):
        ep_ret_str.append(str(round(ep_ret_lst[i], 2)))

    # Print logging statements
    print(flush=True)
    print(f"-------------------- Test #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    for i in range(if_count):
        print(f"Interface {i} Episodic Return: {ep_ret_str[i]}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)


def rollout(policy_lst, mult_if):
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
    
    if_count = len(policy_lst)

    # Rollout until user kills process
    for _ in range(10):
        mult_if.reset()
        state = [mult_if.get_compound_state(i) for i in range(if_count)]
        hidden_state = [policy_lst[i].init_hidden() for i in range(if_count)]

        # number of timesteps so far
        t = 0

        # Logging data
        ep_len = 0  # episodic length
        ep_ret = [0] * if_count
        logits = [None] * if_count
        filtered_logits = [None] * if_count
        action_idx = [None] * if_count

        for _ in range(20):
            t += 1

            # Query deterministic action from policy and run it
            state = [torch.tensor(state[i], dtype=torch.float).view(1, 1, -1) for i in range(if_count)]

            # select the best action
            available_actions_idx = []
            for if_idx in range(if_count):
                available_actions_idx.append([mult_if.get_if(if_idx).get_transition_idx(tr_name, tr_target_if=i)
                                              for tr_name, i in mult_if.get_if(if_idx).available_transitions_with_if()])

                logits[if_idx], hidden_state[if_idx] = policy_lst[if_idx](state[if_idx], hidden_state[if_idx])
                logits[if_idx] = logits[if_idx].view(-1)

                filtered_logits[if_idx] = logits[if_idx][available_actions_idx[if_idx]]

                # choose action with maximal value
                action_idx[if_idx] = torch.argmax(filtered_logits[if_idx])

            real_action = [available_actions_idx[i][action_idx[i].numpy()] for i in range(if_count)]

            reward = [0] * if_count
            checked_interface = [False] * if_count

            # check if two different interfaces tried to communicate with each other
            for i in range(if_count):
                for j in range(i + 1, if_count):
                    if mult_if.check_comm_attempt(i, j, real_action[i], real_action[j]):
                        state[i], state[j], reward[i] = mult_if.step(i, j, real_action[i], real_action[j])
                        reward[j] = reward[i]
                        checked_interface[i] = checked_interface[j] = True

            # handle the other actions (missed communication or local actions)
            for i in range(if_count):
                if not checked_interface[i]:
                    if mult_if.get_if(i).get_transition_by_idx(real_action[i]).is_global():
                        state[i], reward[i] = mult_if.missed_global_step(real_action[i], i)
                    else:
                        state[i], reward[i] = mult_if.local_step(real_action[i], i)
                    checked_interface[i] = True

            # Sum all episodic rewards as we go along
            for i in range(if_count):
                ep_ret[i] += reward[i]

        # Track episodic length
        ep_len = t

        # returns episodic length and return in this iteration
        yield ep_len, ep_ret


def eval_policy(policy_lst, mult_if):
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
    for ep_num, (ep_len, ep_ret_lst) in enumerate(rollout(policy_lst, mult_if)):
        _log_summary(ep_len=ep_len, ep_ret_lst=ep_ret_lst, ep_num=ep_num)
