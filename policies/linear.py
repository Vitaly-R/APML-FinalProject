import numpy as np
from policies.base_policy import Policy
from collections import deque
import random as rnd
from keras.utils import to_categorical
from math import ceil
import random as r

VALUES = 11
LEARNING_RATE = 1e-3
EPSILON = 1.0
BATCH_SIZE = 20
# MAX_BATCH_SIZE = 10
# MIN_BATCH_SIZE = 1
BATCH_THRESHOLD = 300
RADIUS = 2
WINDOW_SIDE_LENGTH = (2 * RADIUS + 1)
# LENGTHS = [(1 + 2 * RADIUS) - abs(2 * (i - RADIUS)) for i in range(WINDOW_SIDE_LENGTH)]
# NUM_ELEMENTS = VALUES * np.sum(LENGTHS)
NUM_ELEMENTS = (WINDOW_SIDE_LENGTH ** 2) * VALUES  # 11 possible values for each of the elements in the window
GAMMA = 0.1


class LinearAgent(Policy):

    def cast_string_args(self, policy_args):
        """
        this function casts arguments passed during policy construction to their proper types/names.
        :param policy_args: an arg -> string value map as received in command line.
        :return: A map of string -> value after casting to useful objects, these will be added as members to the policy
        """
        policy_args['lr'] = float(policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        return policy_args

    def init_run(self):
        """
        this function is called right after the initialization of the agent.
        you may use it to initialize variables that are needed for your policy,
        such as Keras models and so on. you may also use this function
        to load your pickled model and set the variables accordingly, if the
        game uses a saved model and is not a training session.
        """
        self.epsilon = self.__dict__['epsilon']
        self.learning_rate = self.__dict__['lr']
        self.gamma = self.__dict__['gamma']

        self.batch_size = BATCH_SIZE
        self.weights = np.random.random((3, NUM_ELEMENTS)).astype('float32')
        self.memory = deque(maxlen=1000)
        self.epsilon_min = 0.10
        self.epsilon_decay = 0.001 ** (1/(self.game_duration - self.score_scope - self.batch_size))
        # self.learning_rate_min = 1e-5
        self.learning_rate_decay = 0.9999

        # self.exploration_decay = self.epsilon / (self.game_duration - self.score_scope - self.batch_size)

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
        if round < self.batch_size:
            return
        # if round >= 1000:
        #     self.learning_rate *= self.learning_rate_decay

        # update the weights, epsilon and lr
        # self.epsilon = self.epsilon - self.exploration_decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # if self.learning_rate > self.learning_rate_min:
        #     self.learning_rate *= self.learning_rate_decay

        # # train on batch
        # if too_slow:
        #     self.batch_size = max(self.batch_size // 2, MIN_BATCH_SIZE)
        # elif not round % 50:  # to prevent from happening too often
        #     self.batch_size = min(self.batch_size + 1, MAX_BATCH_SIZE)

        # minibatch = rnd.sample(self.memory, self.batch_size)
        minibatch = rnd.choices(self.memory, k=self.batch_size)
        for prev_s, prev_action, reward, new_s in minibatch:
            q_vals = self.weights.dot(new_s)
            max_idx = np.argmax(q_vals)
            max_q_val = q_vals[max_idx]
            # self.weights[max_idx] = (1 - self.learning_rate) * self.weights[max_idx] + self.learning_rate * (reward + self.gamma * max_q_val) * prev_s
            # self.weights[max_idx] += self.learning_rate * (reward + self.gamma * max_q_val - prev_s) * new_s
            self.weights[max_idx] = (1-self.learning_rate) * self.weights[max_idx] + self.learning_rate * (reward + self.gamma * max_q_val ) * prev_s

        # # normalize weights
        # norm = np.linalg.norm(self.weights)
        # if norm:
        #     self.weights /= norm

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
        if prev_state is not None:
            prev_features = self.get_features(prev_state)
            curr_features = self.get_features(new_state)
            self.memory.append([prev_features, prev_action, reward, curr_features])
            if np.random.random() < self.epsilon:
                return self.ACTIONS[np.argmax(self.weights.dot(curr_features))]
        return r.sample(self.ACTIONS, 1)[0]

    def get_features(self, state):
        # get window
        board, head_pos = state[0], state[1][0]
        rows = [i % self.board_size[0] for i in
                range(head_pos[0] - RADIUS + self.board_size[0], head_pos[0] + RADIUS + self.board_size[0] + 1)]
        cols = [i % self.board_size[1] for i in
                range(head_pos[1] - RADIUS + self.board_size[1], head_pos[1] + RADIUS + self.board_size[1] + 1)]
        window = board[rows][:, cols]

        # rotate window, no need to rotate North
        if state[1][1] == 'E':
            window = np.rot90(window, 3)
        elif state[1][1] == 'W':
            window = np.rot90(window, 1)
        elif state[1][1] == 'S':
            window = np.rot90(window, 2)

        # process window and return features
        # window = window.tolist()
        # representation = []
        # for i in range(len(window)):
        #     representation = representation + window[i][RADIUS - ceil(LENGTHS[i] / 2) + 1: RADIUS + ceil(LENGTHS[i] / 2)]
        # representation = np.array(representation)
        # return to_categorical(representation, num_classes=VALUES).reshape(NUM_ELEMENTS)
        return to_categorical(window, num_classes=VALUES).reshape(NUM_ELEMENTS)
