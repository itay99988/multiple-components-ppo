import subprocess
import time

RUN_CMD = ["python", "ppo_experiment.py"]
RUN_EVAL_CMD = ["python", "ppo_experiment.py", "--mode=test", "--actor_model=ppo_actor"]
LOGS_PATH = './logs'


def run_experiment(params):
    # basic params edit
    full_params = BASIC_PARAMS.copy()
    for k, v in params.items():
        full_params[k] = v

    args = RUN_CMD + ["--" + str(key) + "=" + str(param) for key, param in list(full_params.items())]
    eval_args = RUN_EVAL_CMD + ["--" + str(key) + "=" + str(param) for key, param in list(full_params.items())]
    result_train = subprocess.run(args=args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    result_eval = subprocess.run(args=eval_args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)

    output_file_name = f'{LOGS_PATH}/{time.strftime("%d%m%Y-%H%M%S")}.txt'
    with open(output_file_name, 'w') as output_file:
        if result_train.returncode != 0:
            output_file.write(result_train.stderr)
        else:
            output_file.write(result_train.stdout)
            output_file.write("\n")
            output_file.write(result_eval.stdout)


BASIC_PARAMS = {
    'experiment': 'permitted',
    'batch_timesteps': 4000,
    'episode_timesteps': 20,
    'gamma': 0.99,
    'iteration_updates': 10,
    'lr': 1e-2,
    'clip': 0.2,
    'init_tem': 2.5,
    'tem_decay': 1.01,
    'end_cond': 0.99,
    'total_timesteps': 1500000
}

run_experiment({})
run_experiment({})
