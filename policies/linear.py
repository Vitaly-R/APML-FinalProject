from policies.base_policy import Policy
from collections import deque
import numpy as np
import random

MEMORY_SIZE = 300
EPSILON_0 = 7e-2
GAMMA = 85e-2
LEARNING_RATE = 1e-3
BATCH_SIZE = 50
VALUES = 11
RADIUS = 2
WINDOW_SIDE = 2 * RADIUS + 1
ELEMENTS = WINDOW_SIDE ** 2
FEATURES = VALUES * ELEMENTS


class Linear(Policy):

    def cast_string_args(self, policy_args):
        """
        this function casts arguments passed during policy construction to their proper types/names.
        :param policy_args: an arg -> string value map as received in command line.
        :return: A map of string -> value after casting to useful objects, these will be added as members to the policy
        """
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON_0
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['lr'] = float(policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE
        policy_args['bs'] = int(policy_args['bs']) if 'bs' in policy_args else BATCH_SIZE
        return policy_args

    def init_run(self):
        """
        this function is called right after the initialization of the agent.
        you may use it to initialize variables that are needed for your policy,
        such as Keras models and so on. you may also use this function
        to load your pickled model and set the variables accordingly, if the
        game uses a saved model and is not a training session.
        """
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = self.__dict__['epsilon']
        self.gamma = self.__dict__['gamma']
        self.lr = self.__dict__['lr']
        self.bs = self.__dict__['bs']
        self.weights = np.random.uniform(0, 1, (len(self.ACTIONS), FEATURES))

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        the function for learning and improving the policy. it accepts the
        state-action-reward needed to learn from the final move of the game,
        and from that (and other state-action-rewards saved previously) it
        may improve the policy.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
                          This is the final state of the round.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row. you may use this to make your
                        computation time smaller (by lowering the batch size for example).
        """
        if len(self.memory) > 0:
            batch = random.choices(self.memory, k=self.bs)
            for (p_features, p_action, r, n_features) in batch:
                action_index = self.ACTIONS.index(p_action)
                max_future_q = np.max(np.dot(self.weights, n_features))
                prev_q = np.dot(self.weights, p_features)[action_index]
                self.weights[action_index] = self.weights[action_index] + self.lr * (self.gamma * max_future_q + r - prev_q) * p_features

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        the function for choosing an action, given current state.
        it accepts the state-action-reward needed to learn from the previous
        move (which it can save in a data structure for future learning), and
        accepts the new state from which it needs to decide how to act.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row. use this to make your
                        computation time smaller (by lowering the batch size for example)...
        :return: an action (from Policy.Actions) in response to the new_state.
        """
        if round > (self.game_duration - self.score_scope):
            self.epsilon = 0
        new_features = self.process_state(new_state)
        if prev_state is not None:
            prev_features = self.process_state(prev_state)
            self.memory.append((prev_features, prev_action, reward, new_features))
        if np.random.random() < self.epsilon:
            return np.random.choice(self.ACTIONS)
        return self.ACTIONS[np.argmax(np.dot(self.weights, new_features))]

    def process_state(self, state):
        (board, (pos, direction)) = state
        rows = [i % self.board_size[0] for i in range(pos[0] - RADIUS, pos[0] + RADIUS + 1)]
        cols = [i % self.board_size[1] for i in range(pos[1] - RADIUS, pos[1] + RADIUS + 1)]
        window = board[rows][:, cols]
        if direction == 'N':
            return self.to_features(window)
        elif direction == 'E':
            return self.to_features(np.rot90(window, 1))
        elif direction == 'S':
            return self.to_features(np.rot90(window, 2))
        elif direction == 'W':
            return self.to_features(np.rot90(window, 3))

    @staticmethod
    def to_features(window):
        features = np.zeros(FEATURES)
        flattened = np.reshape(window, ELEMENTS)
        for i in range(flattened.shape[0]):
            features[i * VALUES + flattened[i] + 1] = 1  # the + 1 in the index is to offset the values from [-1, 9] to [0, 10]
        return features
