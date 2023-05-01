from process import Process, SUCCESS, FAIL, RANDOM


# a distributed system is a compound of processes (of class "Process").
class DistSystem(object):
    def __init__(self, name, processes):
        self.name = name
        self.processes = processes

    # reinitialize all processes in the system
    def reset(self):
        for pr in self.processes:
            pr.reset()

    # returns the process with that name
    def get_process(self, name):
        return next(proc for proc in self.processes if name == proc.name)

    # adds a new process to the system
    def add_process(self, process):
        self.processes.append(process)


# This class inherits the properties of DistSystem. It is a specific case in which there are only three processes.
class TripleInterface(DistSystem):
    def __init__(self, name, if1: Process, if2: Process, if3: Process, history_len=1):
        super().__init__(name, [if1, if2, if3])
        self.ifs = [if1, if2, if3]
        self.history_len = history_len
        self.if_state_history = [None, None, None]
        self.if_execution = [[], [], []]
        self.if_count = len(self.ifs)

    def get_if(self, if_idx):
        return self.ifs[if_idx]

    def reset(self):
        super().reset()
        self.init_state_history()
        self.init_executions()

    def init_state_history(self):
        for i in range(self.if_count):
            self.if_state_history[i] = []
            for _ in range(self.history_len):
                if_partial_state = self.ifs[i].get_rnn_input(None)
                self.if_state_history[i].append(if_partial_state)

    def get_compound_state(self, if_idx):
        compound_state_lst = self.if_state_history[if_idx][-self.history_len:]

        return [item for sublist in compound_state_lst for item in sublist]

    def init_executions(self):
        for i in range(self.if_count):
            self.if_execution[i] = []

    def step(self, if1_idx, if2_idx, if1_tr_idx, if2_tr_idx):
        if1_tr = self.get_if(if1_idx).transitions[if1_tr_idx].name
        if1_tr_target_if = self.get_if(if1_idx).transitions[if1_tr_idx].target_if_idx
        if2_tr = self.get_if(if2_idx).transitions[if2_tr_idx].name
        if2_tr_target_if = self.get_if(if2_idx).transitions[if2_tr_idx].target_if_idx

        # successful communication action
        if if1_tr == if2_tr:
            full_if1_tr = self.get_if(if1_idx).copy_transition_w_status(if1_tr, if1_tr_target_if, status=SUCCESS)
            full_if2_tr = self.get_if(if2_idx).copy_transition_w_status(if2_tr, if2_tr_target_if, status=SUCCESS)

            self.if_execution[if1_idx].append(self.get_if(if1_idx).copy_transition_w_status(if1_tr,
                                                                                            if1_tr_target_if,
                                                                                            status=SUCCESS))
            self.if_execution[if2_idx].append(self.get_if(if2_idx).copy_transition_w_status(if2_tr,
                                                                                            if2_tr_target_if,
                                                                                            status=SUCCESS))

            # trigger the transition for both interfaces
            reward = max(self.get_if(if1_idx).get_transition_reward(if1_tr),
                         self.get_if(if2_idx).get_transition_reward(if2_tr))
            self.get_if(if1_idx).trigger_transition(if1_tr)
            self.get_if(if2_idx).trigger_transition(if2_tr)
        else:
            full_if1_tr = self.get_if(if1_idx).copy_transition_w_status(if1_tr, if1_tr_target_if, status=FAIL)
            full_if2_tr = self.get_if(if2_idx).copy_transition_w_status(if2_tr, if2_tr_target_if,  status=FAIL)

            self.if_execution[if1_idx].append(self.get_if(if1_idx).copy_transition_w_status(if1_tr,
                                                                                            if1_tr_target_if,
                                                                                            status=FAIL))
            self.if_execution[if2_idx].append(self.get_if(if2_idx).copy_transition_w_status(if2_tr,
                                                                                            if2_tr_target_if,
                                                                                            status=FAIL))

            if1_rnd_tr = self.get_if(if1_idx).get_random_transition()
            if2_rnd_tr = self.get_if(if2_idx).get_random_transition()

            # trigger the randomly chosen transitions for both interfaces
            reward = -1 * max(self.get_if(if1_idx).get_transition_reward(if1_tr),
                              self.get_if(if2_idx).get_transition_reward(if2_tr))
            self.get_if(if1_idx).trigger_transition(if1_rnd_tr)
            self.get_if(if2_idx).trigger_transition(if2_rnd_tr)


        if1_partial_state = self.get_if(if1_idx).get_rnn_input(full_if1_tr)
        if2_partial_state = self.get_if(if2_idx).get_rnn_input(full_if2_tr)

        # building compound states for both interfaces
        self.if_state_history[if1_idx].append(if1_partial_state)
        self.if_state_history[if2_idx].append(if2_partial_state)

        if1_next_state = self.get_compound_state(if1_idx)
        if2_next_state = self.get_compound_state(if2_idx)

        return if1_next_state, if2_next_state, reward

    def local_step(self, if_tr_idx, if_idx):
        if_tr = self.get_if(if_idx).transitions[if_tr_idx].name
        full_if_tr = self.get_if(if_idx).copy_transition_w_status(if_tr, 9, status=SUCCESS)
        self.if_execution[if_idx].append(self.get_if(if_idx).copy_transition_w_status(if_tr, 9, status=SUCCESS))

        # trigger the transition for the interface
        reward = self.get_if(if_idx).get_transition_reward(if_tr)
        self.get_if(if_idx).trigger_transition(if_tr)

        if_partial_state = self.get_if(if_idx).get_rnn_input(full_if_tr)

        # building compound states for both interfaces
        self.if_state_history[if_idx].append(if_partial_state)
        if_next_state = self.get_compound_state(if_idx)

        return if_next_state, reward

    def missed_global_step(self, if_tr_idx, if_idx):
        if_tr = self.get_if(if_idx).transitions[if_tr_idx].name
        target_if = self.get_if(if_idx).transitions[if_tr_idx].target_if_idx

        full_if_tr = self.get_if(if_idx).copy_transition_w_status(if_tr, target_if, status=FAIL)
        self.if_execution[if_idx].append(self.get_if(if_idx).copy_transition_w_status(if_tr, target_if, status=FAIL))

        if_partial_state = self.get_if(if_idx).get_rnn_input(full_if_tr)

        # building compound state for the interface
        self.if_state_history[if_idx].append(if_partial_state)
        if_next_state = self.get_compound_state(if_idx)

        # assign a reward of zero for a missed communication action
        reward = 0

        return if_next_state, reward

    def check_comm_attempt(self, if1_idx, if2_idx, if1_tr_idx, if2_tr_idx):
        if1_full_tr = self.get_if(if1_idx).get_transition_by_idx(if1_tr_idx)
        if2_full_tr = self.get_if(if2_idx).get_transition_by_idx(if2_tr_idx)

        if if1_full_tr.is_global() and if2_full_tr.is_global():
            if if1_full_tr.get_target_if() == if2_idx and if2_full_tr.get_target_if() == if1_idx:
                return True

        return False

    def get_exploration_status(self, if_idx):
        return self.get_if(if_idx).get_exploration_status()

    def set_exploration_status(self, if_idx, status):
        self.get_if(if_idx).set_exploration_status(status)

    def invert_exploration_status(self, if_idx):
        status = self.get_exploration_status(if_idx)
        self.set_exploration_status(if_idx, not status)
