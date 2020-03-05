from policies.base_policy import Policy
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from collections import deque
from math import ceil
import numpy as np
import random as r

VALUES = 11
LEARNING_RATE = 1e-3
BATCH_SIZE = 20
MAX_BATCH_SIZE = 30
MIN_BATCH_SIZE = 10
# BATCH_THRESHOLD = 20 * BATCH_SIZE
# BATCH_THRESHOLD = 100
RADIUS = 4
WINDOW_SIDE_LENGTH = 2 * RADIUS + 1
LENGTHS = [(1 + 2 * RADIUS) - abs(2 * (i - RADIUS)) for i in range(WINDOW_SIDE_LENGTH)]
NUM_ELEMENTS = VALUES * np.sum(LENGTHS)
GAMMA = 0.85
INITIAL_EPSILON = 0.1
EPSILON_DECAY = 0.999
# EPSILON_DECAY_ROUND = BATCH_THRESHOLD
# EPSILON_MIN = 0.05
DENSE_SIZE = 32


class CustomPolicy(Policy):
    def cast_string_args(self, policy_args):
        """
        this function casts arguments passed during policy construction to their proper types/names.
        :param policy_args: an arg -> string value map as received in command line.
        :return: A map of string -> value after casting to useful objects, these will be added as members to the policy
        """
        policy_args['lr'] = float(policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else INITIAL_EPSILON
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['dense_size'] = int(policy_args['dense_size']) if 'dense_size' in policy_args else DENSE_SIZE
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
        self.lr = self.__dict__['lr']
        self.gamma = self.__dict__['gamma']
        self.dense_size = self.__dict__['dense_size']

        print('creating simple network')
        self.model = self.create_model(self.dense_size)
        self.target_model = self.create_model(self.dense_size)
        self.model.predict(np.zeros((1, NUM_ELEMENTS)))

        self.batch_size = BATCH_SIZE
        self.exploration_decay = self.epsilon / (self.game_duration - self.score_scope - self.batch_size)
        self.memory = deque(maxlen=1000)
        self.t = 0.5  # a learning rate for the weights of the target model in relation to the main model

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

        if too_slow:
            self.batch_size = max(self.batch_size // 2, MIN_BATCH_SIZE)
        elif not round % 50:
            self.batch_size = min(self.batch_size + 1, MAX_BATCH_SIZE)

        # self.epsilon = self.epsilon - self.exploration_decay
        self.epsilon *= EPSILON_DECAY

        data = list()
        targets = list()
        samples = r.choices(self.memory, k=self.batch_size)
        for (prev_features, action, reward, curr_features) in samples:
            target = self.target_model.predict(prev_features)  # q-values of the target model for each action
            q_future = max(self.target_model.predict(curr_features)[0])  # the max q-value possible from the next state
            # print("*************** q values: " + str(target))
            target[0][self.ACTIONS.index(
                action)] = reward + self.gamma * q_future  # the target q-value we want our model to achieve for the specific action
            targets.append(target[0].tolist())
            data.append(prev_features[0].tolist())
        self.model.fit(np.array(data), np.array(targets), epochs=1,
                       verbose=0)  # training and updating the weights of the main model for the target activation.

        # updating the weights of the target model in relation to the main model.
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.t + target_weights[i] * (1 - self.t)
        self.target_model.set_weights(target_weights)

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
                return self.ACTIONS[np.argmax(self.model.predict(curr_features))]
        return r.sample(self.ACTIONS, 1)[0]

    def create_model(self, dense_size=128):
        model = Sequential()
        model.add(Dense(dense_size, activation='tanh', kernel_regularizer=l2()))
        model.add(Dense(len(self.ACTIONS), kernel_regularizer=l2()))
        model.compile(optimizer=Adam(lr=self.lr), loss='mean_squared_error')
        return model

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
            window = np.rot90(window, 1)
        elif state[1][1] == 'W':
            window = np.rot90(window, 3)
        elif state[1][1] == 'S':
            window = np.rot90(window, 2)

        # process window and return features
        window = window.tolist()
        representation = []
        for i in range(len(window)):
            representation = representation + window[i][RADIUS - ceil(LENGTHS[i] / 2) + 1: RADIUS + ceil(LENGTHS[i] / 2)]
        representation = np.array(representation)
        return to_categorical(representation, num_classes=VALUES).reshape((1, NUM_ELEMENTS))


