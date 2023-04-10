"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py.
"""
import torch


def _log_summary(ep_len, ep_ret1, ep_ret2, ep_ret3, ep_num):
    """
        Print to stdout what we've logged so far in the most recent episode.
        Parameters:
            None
        Return:
            None
    """
    # Round decimal places for more aesthetic logging messages
    failure_rate1 = str(round((ep_len - ep_ret1) * (50 / ep_len), 2))
    failure_rate2 = str(round((ep_len - ep_ret2) * (50 / ep_len), 2))
    failure_rate3 = str(round((ep_len - ep_ret3) * (50 / ep_len), 2))
    ep_len = str(round(ep_len, 2))
    ep_ret1 = str(round(ep_ret1, 2))
    ep_ret2 = str(round(ep_ret2, 2))
    ep_ret3 = str(round(ep_ret3, 2))

    # Print logging statements
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Interface 1 Episodic Return: {ep_ret1}", flush=True)
    print(f"Interface 2 Episodic Return: {ep_ret2}", flush=True)
    print(f"Interface 3 Episodic Return: {ep_ret3}", flush=True)
    print("Interface 1 Failure Rate: {}%".format(failure_rate1))
    print("Interface 2 Failure Rate: {}%".format(failure_rate2))
    print("Interface 3 Failure Rate: {}%".format(failure_rate3))
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)


def rollout(policy1, policy2, policy3, triple_if):
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
        triple_if.reset()
        state = [triple_if.get_compound_state(0), triple_if.get_compound_state(1), triple_if.get_compound_state(2)]
        hidden_state = [policy1.init_hidden(), policy2.init_hidden(), policy3.init_hidden()]

        # number of timesteps so far
        t = 0

        # Logging data
        ep_len = 0  # episodic length
        ep_ret1 = 0  # episodic return of interface1
        ep_ret2 = 0  # episodic return of interface2
        ep_ret3 = 0  # episodic return of interface3

        for _ in range(20):
            t += 1

            # Query deterministic action from policy and run it
            state = [torch.tensor(state[0], dtype=torch.float).view(1, 1, -1),
                     torch.tensor(state[1], dtype=torch.float).view(1, 1, -1),
                     torch.tensor(state[2], dtype=torch.float).view(1, 1, -1)
                     ]


            # select the best action
            available_actions_idx1 = [triple_if.get_if(0).get_transition_idx(tr_name, tr_target_if=i) for tr_name, i in
                                      triple_if.get_if(0).available_transitions_with_if()]
            available_actions_idx2 = [triple_if.get_if(1).get_transition_idx(tr_name, tr_target_if=i) for tr_name, i in
                                      triple_if.get_if(1).available_transitions_with_if()]
            available_actions_idx3 = [triple_if.get_if(2).get_transition_idx(tr_name, tr_target_if=i) for tr_name, i in
                                      triple_if.get_if(2).available_transitions_with_if()]


            logits1, hidden_state[0] = policy1(state[0], hidden_state[0])
            logits1 = logits1.view(-1)
            logits2, hidden_state[1] = policy2(state[1], hidden_state[1])
            logits2 = logits2.view(-1)
            logits3, hidden_state[2] = policy3(state[2], hidden_state[2])
            logits3 = logits3.view(-1)

            filtered_logits1 = logits1[available_actions_idx1]
            filtered_logits2 = logits2[available_actions_idx2]
            filtered_logits3 = logits3[available_actions_idx3]

            # choose based on probability
            # dist = Categorical(logits=filtered_logits)
            # action_idx = dist.sample()

            # choose action with maximal value
            action_idx1 = torch.argmax(filtered_logits1)
            action_idx2 = torch.argmax(filtered_logits2)
            action_idx3 = torch.argmax(filtered_logits3)

            real_action = [available_actions_idx1[action_idx1.numpy()],
                           available_actions_idx2[action_idx2.numpy()],
                           available_actions_idx3[action_idx3.numpy()]
                           ]

            reward = [0, 0, 0]
            checked_interface = [False, False, False]

            # check if two different interfaces tried to communicate with each other
            if triple_if.check_comm_attempt(0, 1, real_action[0], real_action[1]):
                state[0], state[1], reward[0] = triple_if.step(0, 1, real_action[0], real_action[1])
                reward[1] = reward[0]
                checked_interface[0] = checked_interface[1] = True
            elif triple_if.check_comm_attempt(0, 2, real_action[0], real_action[2]):
                state[0], state[2], reward[0] = triple_if.step(0, 2, real_action[0], real_action[2])
                reward[2] = reward[0]
                checked_interface[0] = checked_interface[2] = True
            elif triple_if.check_comm_attempt(1, 2, real_action[1], real_action[2]):
                state[1], state[2], reward[1] = triple_if.step(1, 2, real_action[1], real_action[2])
                reward[2] = reward[1]
                checked_interface[1] = checked_interface[2] = True

            # handle the other actions (missed communication or local actions)
            for i in range(triple_if.if_count):
                if not checked_interface[i]:
                    if triple_if.get_if(i).get_transition_by_idx(real_action[i]).is_global():
                        state[i], reward[i] = triple_if.missed_global_step(real_action[i], i)
                    else:
                        state[i], reward[i] = triple_if.local_step(real_action[i], i)
                    checked_interface[i] = True

            # Sum all episodic rewards as we go along
            ep_ret1 += reward[0]
            ep_ret2 += reward[1]
            ep_ret3 += reward[2]

        # Track episodic length
        ep_len = t

        # returns episodic length and return in this iteration
        yield ep_len, ep_ret1, ep_ret2, ep_ret3


def eval_policy(policy1, policy2, policy3, triple_if):
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
    for ep_num, (ep_len, ep_ret1, ep_ret2, ep_ret3) in enumerate(rollout(policy1, policy2, policy3, triple_if)):
        _log_summary(ep_len=ep_len, ep_ret1=ep_ret1, ep_ret2=ep_ret2, ep_ret3=ep_ret3, ep_num=ep_num)
