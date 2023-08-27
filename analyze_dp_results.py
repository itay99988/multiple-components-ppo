def analyze_joint_policy(policy_lst, eat_op_str):
    timestep_count = len(policy_lst[0])
    p_policies = policy_lst[::2]

    for step in range(timestep_count):
        first_one_to_eat = True
        for i in range(len(p_policies)):
            if p_policies[i][step][0] == eat_op_str:
                if first_one_to_eat:
                    print(f'Timestep {step + 1}:')
                    first_one_to_eat = False
                print(f'Philosopher {i+1} ate.')
