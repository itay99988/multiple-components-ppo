import random

SUCCESS = "(Success)"
FAIL = "(Failure)"
RANDOM = "(Random)"


# A transition class (transition of a process). contains name, source state, target state, and a status (success,
# failure or random)
class Transition:
    def __init__(self, name, source_state, target_state, reward=0, global_action=True, status=None):
        self.name = name
        self.source_state = source_state
        self.target_state = target_state
        self.reward = reward
        self.global_action = global_action
        self.status = status

    def copy(self, status):
        return Transition(self.name, self.source_state, self.target_state, self.reward, self.global_action, status)

    def is_global(self):
        return self.global_action

    def get_reward(self):
        return self.reward

    # a string representation of a transition - to be used in the execution report
    def __str__(self):
        return "{0} ---------{1}{2}---------> {3}".format(self.source_state, self.name, self.status, self.target_state)


# A process class - can be either the system or the environment (a finite and deterministic automaton)
class Process:
    def __init__(self, name, states=[], transitions=[], initial_state=None):
        self.name = name
        self.states = states  # list of names of the states.
        self.initial_state = initial_state
        self.current_state = initial_state
        self.transitions = transitions  # transitions that are specific to the process.
        self.exploration = True

    def add_state(self, name):
        self.states.append(name)
        if self.current_state is None:
            self.initial_state = name
            self.current_state = name

    # return the current state of the process
    def get_current_state(self):
        return self.current_state

    # sets a new state as the current state (in case of triggering a transition for example)
    def set_current_state(self, name):
        if name in self.states:
            self.current_state = name

    def set_exploration_status(self, status):
        self.exploration = status

    def get_exploration_status(self):
        return self.exploration

    # adds a new transition to the process
    def add_transition(self, name, source, target, reward=0, global_action=True):
        self.transitions.append(Transition(name, source, target, reward=reward, global_action=global_action))

    # returns to the initial state of the transition
    def reset(self):
        self.set_current_state(self.initial_state)

    # returns the correct transition object according to its name and its source state
    # (the source state is enough in order to distinguish two transitions with the same name)
    def get_transition(self, tr_name, source_state=None):
        if source_state is None:
            source_state = self.current_state
        possible_tr = (tr for tr in self.transitions if tr.name == tr_name and tr.source_state == source_state)
        return next(possible_tr)

    # returns the index of a specific transition from the transition list. important for the loss function calculation.
    def get_transition_idx(self, tr_name, source_state=None):
        if source_state is None:
            source_state = self.current_state
        possible_idx = (i for i, tr in enumerate(self.transitions) if tr.name == tr_name and tr.source_state == source_state)
        return next(possible_idx)

    def get_transition_by_idx(self, tr_idx):
        return self.transitions[tr_idx]

    def get_transition_reward(self, tr_name):
        return self.get_transition(tr_name).get_reward()

    # copies an entire transition object, with a new status
    def copy_transition_w_status(self, tr_name, source_state=None, status=None):
        if source_state is None:
            source_state = self.current_state
        orig_tr = self.get_transition(tr_name, source_state)
        return orig_tr.copy(status)

    # returns a set of names of transitions that can be triggered in the current state.
    def available_transitions(self):
        available = []
        for tr in self.transitions:
            if tr.source_state == self.current_state:
                available.append(tr.name)
        return available

    # switches the process' state according to the transition tr_name.
    def trigger_transition(self, tr_name):
        try:
            possible_next_states = (tr.target_state for tr in self.transitions
                                    if tr.name == tr_name and tr.source_state == self.current_state)
            self.current_state = next(possible_next_states)

        except StopIteration:
            print("No transition named", tr_name, "from state", self.current_state)

    # chooses a random available state uniformly
    def get_random_transition(self):
        return random.choice(self.available_transitions())

    # checks if a certain transition is currently enabled
    def is_transition_enabled(self, tr_name):
        return tr_name in self.available_transitions()

    # gets the last system's transition (and its status - success or fail) and returns the input of the controller
    # this input reprsents a flattened matrix (transitions X states) with only one non zero cell. the location of the
    # non zero cell depends of the last transition and the current state. the non zero cell will be "1" if the last
    # transition was successful and "-1" otherwise.
    def get_rnn_input(self, tr: Transition):
        vec = [0] * (len(self.transitions) * len(self.states))

        if tr is not None:
            state_idx = self.states.index(self.current_state)
            transition_idx = self.get_transition_idx(tr.name, tr.source_state)

            # "-1" in case of failure and "1" otherwise
            if tr.status == FAIL:
                vec[len(self.states) * transition_idx + state_idx] = -1
            else:
                vec[len(self.states) * transition_idx + state_idx] = 1

        return vec

    # returns the name of the predicted transition according to the network output
    # there are two different methods to infer the next transition according the network output:
    # 1. randomly choose a transition according the softmax distribution of the network
    # 2. always choose the most probable transition (argmax)
    def get_predicted_transition(self, rnn_output, method="by_tr_distribution", debug_prob=False, debug_file="sys1_probs"):
        available_transitions = self.available_transitions()
        # leave only the possible transitions
        distribution = [rnn_output[:, i] for i, tr in enumerate(self.transitions)
                        if tr.name in available_transitions
                        and tr.source_state == self.current_state]

        if debug_prob:
            with open(debug_file, 'a') as f:
                print(distribution, file=f)

        # randomly choose a transition according the softmax distribution of the network
        if method == "by_tr_distribution":
            predicted_transition = random.choices(available_transitions, distribution)[0]

        # always choose the most probable transition at the time
        elif method == "argmax":
            i2v = lambda i: distribution[i]
            idx_max = max(range(len(distribution)), key=i2v)
            predicted_transition = available_transitions[idx_max]

        else:
            raise NameError("Incorrect method")

        # return the chosen transition
        return predicted_transition
