from policies.base_policy import Policy
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from math import ceil
import numpy as np
import random

MEMORY_SIZE = 500
EPSILON_0 = 0.1
DECAY_RATE = 0.99
GAMMA = 85e-2
LEARNING_RATE = 1e-3
BATCH_SIZE = 15
VALUES = 11
RADIUS = 2
WINDOW_SIDE = 2 * RADIUS + 1
LENGTHS = [(1 + 2 * RADIUS) - abs(2 * (i - RADIUS)) for i in range(WINDOW_SIDE)]
ELEMENTS = np.sum(LENGTHS)
FEATURES = VALUES * ELEMENTS


class Custom(Policy):

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
        self.epsilon = self.__dict__['epsilon']
        self.gamma = self.__dict__['gamma']
        self.lr = self.__dict__['lr']
        self.bs = self.__dict__['bs']

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = self.get_model()
        init_in = np.zeros((1, FEATURES))
        init_prediction = self.model.predict(init_in)
        self.model.fit(init_in, init_prediction)

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
        self.epsilon = self.epsilon * DECAY_RATE if not round % 500 else self.epsilon
        if len(self.memory) > 0:
            samples = random.choices(self.memory, k=self.bs)
            p_features_batch = np.zeros((self.bs, FEATURES))
            actions = np.zeros(self.bs, dtype=int)
            p_rewards = np.zeros(self.bs)
            n_features_batch = np.zeros((self.bs, FEATURES))
            for i, (p_features, p_action, r, n_features) in enumerate(samples):
                p_features_batch[i] = p_features
                actions[i] = self.ACTIONS.index(p_action)
                p_rewards[i] = r
                n_features_batch[i] = n_features
            discounted_max_future_qs = self.gamma * np.max(self.model.predict(n_features_batch), axis=1)
            prev_states_predictions = self.model.predict(p_features_batch)
            prev_states_predictions[:, actions] = p_rewards + discounted_max_future_qs
            self.model.fit(p_features_batch, prev_states_predictions, epochs=1, verbose=0)

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
        if (self.game_duration - self.score_scope) < round:
            self.epsilon = 0
        new_features = self.process_state(new_state)
        if prev_state is not None:
            prev_features = self.process_state(prev_state)
            self.memory.append((prev_features.tolist(), prev_action, reward, new_features.tolist()))
            # self.memory.append((prev_features, prev_action, reward, new_features))
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.ACTIONS)
        return self.ACTIONS[np.argmax(self.model.predict(new_features[np.newaxis, ...]))]

    def get_model(self):
        """
        return the model of the custom agent
        """
        model = Sequential()
        model.add(Dense(16, activation='relu'))
        model.add(Dense(len(self.ACTIONS), activation='linear'))
        model.compile(Adam(lr=self.lr), 'mean_squared_error')
        return model

    def process_state(self, state):
        """
        process the state by getting a smaller window, rotating it, and converting it to a rhombus represented as a
        feature vector
        """
        (board, (pos, direction)) = state
        rows = [i % self.board_size[0] for i in range(pos[0] - RADIUS, pos[0] + RADIUS + 1)]
        cols = [i % self.board_size[1] for i in range(pos[1] - RADIUS, pos[1] + RADIUS + 1)]
        window = board[rows][:, cols]
        if direction == 'E':
            window = np.rot90(window, 1)
        elif direction == 'S':
            window = np.rot90(window, 2)
        elif direction == 'W':
            window = np.rot90(window, 3)

        window = window.tolist()
        flattened = []
        for i in range(len(window)):
            flattened = flattened + window[i][RADIUS - ceil(LENGTHS[i] / 2) + 1: RADIUS + ceil(LENGTHS[i] / 2)]
        for i in range(len(flattened)):
            flattened[i] += 1  # in order to offset the range from [-1, 9] to [0, 10]

        return to_categorical(flattened, num_classes=VALUES).reshape(FEATURES)
