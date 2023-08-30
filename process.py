import random

SUCCESS = "(Success)"
FAIL = "(Failure)"
RANDOM = "(Random)"


# A transition class (transition of a process). contains name, source state, target state, and a status (success,
# failure or random)
class Transition:
    def __init__(self, name, source_state, target_state, target_if_idx, reward=0, global_action=True, status=None):
        self.name = name
        self.source_state = source_state
        self.target_state = target_state
        self.target_if_idx = target_if_idx
        self.reward = reward
        self.global_action = global_action
        self.status = status

    def copy(self, status):
        return Transition(self.name, self.source_state, self.target_state, self.target_if_idx,
                          self.reward, self.global_action, status)

    def is_global(self):
        return self.global_action

    def get_reward(self):
        return self.reward

    def get_target_if(self):
        return self.target_if_idx

    # a string representation of a transition - to be used in the execution report
    def __str__(self):
        return "To interface {4}:{0} ---------{1}{2}---------> {3}".format(self.source_state, self.name, self.status,
                                                                           self.target_state, self.target_if_idx)


# A process class - can be either the system or the environment (a finite and deterministic automaton)
class Process:
    def __init__(self, name, states=[], transitions=[], initial_state=None, suspend_func=None):
        self.name = name
        self.states = states  # list of names of the states.
        self.initial_state = initial_state
        self.current_state = initial_state
        self.transitions = transitions  # transitions that are specific to the process.
        self.exploration = True
        self.select_random = False
        self.suspend_func = suspend_func

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
    def add_transition(self, name, source, target, target_if, reward=0, global_action=True):
        self.transitions.append(Transition(name, source, target, target_if, reward=reward, global_action=global_action))

    # returns to the initial state of the transition
    def reset(self):
        self.set_current_state(self.initial_state)

    # returns the correct transition object according to its name and its source state
    # (the source state is enough in order to distinguish two transitions with the same name)
    def get_transition(self, tr_name, tr_target_if=None, source_state=None):
        if source_state is None:
            source_state = self.current_state
        if tr_target_if is None:
            possible_tr = (tr for tr in self.transitions if tr.name == tr_name and tr.source_state == source_state)
        else:
            possible_tr = (tr for tr in self.transitions if tr.name == tr_name and tr.target_if_idx == tr_target_if and
                           tr.source_state == source_state)
        return next(possible_tr)

    # returns the index of a specific transition from the transition list. important for the loss function calculation.
    def get_transition_idx(self, tr_name, tr_target_if=None, source_state=None):
        if source_state is None:
            source_state = self.current_state
        if tr_target_if is None:
            possible_idx = (i for i, tr in enumerate(self.transitions) if tr.name == tr_name and
                            tr.source_state == source_state)
        else:
            possible_idx = (i for i, tr in enumerate(self.transitions) if tr.name == tr_name and
                            tr.target_if_idx == tr_target_if and tr.source_state == source_state)
        return next(possible_idx)

    def get_transition_by_idx(self, tr_idx):
        return self.transitions[tr_idx]

    def get_transition_reward(self, tr_name):
        return self.get_transition(tr_name).get_reward()

    # copies an entire transition object, with a new status
    def copy_transition_w_status(self, tr_name, target_if, source_state=None, status=None):
        if source_state is None:
            source_state = self.current_state
        orig_tr = self.get_transition(tr_name, target_if, source_state)
        return orig_tr.copy(status)

    # returns a set of names of transitions that can be triggered in the current state.
    def available_transitions(self):
        available = []
        for tr in self.transitions:
            if tr.source_state == self.current_state:
                available.append(tr.name)
        return available

    # returns a set of names of transitions that can be triggered in the current state, with their target interface idx.
    def available_transitions_with_if(self):
        available = []
        for tr in self.transitions:
            if tr.source_state == self.current_state:
                available.append((tr.name, tr.target_if_idx))
        return available

    def available_local_transitions(self):
        available = []
        for tr in self.transitions:
            if tr.source_state == self.current_state and not tr.is_global():
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

    def get_random_local_transition(self):
        local_tr_lst = self.available_local_transitions()

        if not local_tr_lst:
            return None
        return random.choice(local_tr_lst)

    def get_suspend_lst(self, ep_len):
        if self.suspend_func is None:
            return []
        return self.suspend_func(ep_len)

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
            transition_idx = self.get_transition_idx(tr.name, tr_target_if=tr.target_if_idx,
                                                     source_state=tr.source_state)

            # "-1" in case of failure and "1" otherwise
            if tr.status == FAIL:
                vec[len(self.states) * transition_idx + state_idx] = -1
            else:
                vec[len(self.states) * transition_idx + state_idx] = 1

        return vec
