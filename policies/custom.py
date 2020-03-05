from policies.base_policy import Policy
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from collections import deque
import numpy as np
import random as r

VALUES = 11
LEARNING_RATE = 1e-3
BATCH_SIZE = 15
MAX_BATCH_SIZE = 30
MIN_BATCH_SIZE = 1
# BATCH_THRESHOLD = 20 * BATCH_SIZE
BATCH_THRESHOLD = 100
RADIUS = 3
GAMMA = 0.1
INITIAL_EPSILON = 0.1
# EPSILON_DECAY = 0.999
# EPSILON_DECAY_ROUND = BATCH_THRESHOLD
# EPSILON_MIN = 0.05
WINDOW_SIDE_LENGTH = 2 * RADIUS + 1
NUM_ELEMENTS = WINDOW_SIDE_LENGTH ** 2
MODEL = 0  # An indicator of weather to use a convolutional model, used solely for testing


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
        policy_args['model'] = int(policy_args['model']) if 'model' in policy_args else MODEL
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
        if self.__dict__['model'] == 0:
            print('creating dense network')
            self.model = self.create_model_1()
            self.target_model = self.create_model_1()
            self.model.predict(np.zeros((1, NUM_ELEMENTS)))
            self.__process_state = self.__process_state_1
        elif self.__dict__['model'] == 1:
            print('creating convolutional network')
            self.model = self.create_model_2()
            self.target_model = self.create_model_2()
            self.model.predict(np.zeros((1, WINDOW_SIDE_LENGTH, WINDOW_SIDE_LENGTH, 1)))
            self.__process_state = self.__process_state_2
        elif self.__dict__['model'] == 2:
            print('creating dense network')
            self.model = self.create_model_4()
            self.target_model = self.create_model_4()
            self.model.predict(np.zeros((1, NUM_ELEMENTS)))
            self.__process_state = self.__process_state_4
        else:
            print('creating simple network')
            self.model = self.create_model_3()
            self.target_model = self.create_model_3()
            self.model.predict(np.zeros((1, NUM_ELEMENTS * VALUES)))
            self.__process_state = self.__process_state_5
        self.exploration_decay = self.epsilon / (self.game_duration - self.score_scope)
        self.memory = deque(maxlen=BATCH_THRESHOLD * 5)
        self.batch_size = BATCH_SIZE
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

        self.epsilon = self.epsilon - self.exploration_decay

        samples = r.sample(self.memory, self.batch_size)
        data = list()
        targets = list()
        for (state, action, reward, next_state) in samples:
            target = self.target_model.predict(state)  # q-values of the target model for each action
            q_future = max(self.target_model.predict(next_state)[0])  # the max q-value possible from the next state
            # print("*************** q values: " + str(target))
            target[0][self.ACTIONS.index(
                action)] = reward + self.gamma * q_future  # the target q-value we want our model to achieve for the specific action
            targets.append(target[0].tolist())
            data.append(state[0].tolist())
        self.model.fit(np.array(data), np.array(targets), epochs=1,
                       verbose=0)  # training and updating the weights of the main model for the target activation.

        # if not round % 50:
        # updating the weights of the target model in relation to the main model.
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.t + target_weights[i] * (1 - self.t)
        self.target_model.set_weights(target_weights)

        # # if round >= EPSILON_DECAY_ROUND:
        # #     # self.epsilon = EPSILON_MIN if self.epsilon <= EPSILON_MIN else self.epsilon - self.exploration_decay
        # #
        #
        # if round > BATCH_THRESHOLD:
        #     # # In order for the model not to start training too early, we wait until enough states are saved.
        #     # if too_slow:
        #     #     self.batch_size = max(self.batch_size // 2, MIN_BATCH_SIZE)
        #     # elif not round % 100:  # to prevent from happening too often
        #     #     self.batch_size = min(self.batch_size + 1, MAX_BATCH_SIZE)
        #     # # print("****************** batch size: " + str(self.batch_size))
        #
        #     samples = r.sample(self.memory, self.batch_size)
        #     data = list()
        #     targets = list()
        #     for (state, action, reward, next_state) in samples:
        #         target = self.target_model.predict(state)  # q-values of the target model for each action
        #         q_future = max(self.target_model.predict(next_state)[0])  # the max q-value possible from the next state
        #         # print("*************** q values: " + str(target))
        #         target[0][self.ACTIONS.index(
        #             action)] = reward + self.gamma * q_future  # the target q-value we want our model to achieve for the specific action
        #         targets.append(target[0].tolist())
        #         data.append(state[0].tolist())
        #     self.model.fit(np.array(data), np.array(targets), epochs=1,
        #                    verbose=0)  # training and updating the weights of the main model for the target activation.
        #
        #     # if not round % 50:
        #     # updating the weights of the target model in relation to the main model.
        #     weights = self.model.get_weights()
        #     target_weights = self.target_model.get_weights()
        #     for i in range(len(target_weights)):
        #         target_weights[i] = weights[i] * self.t + target_weights[i] * (1 - self.t)
        #     self.target_model.set_weights(target_weights)

        # # TODO: testing if the weights get too high
        # if not round % 500 and round > 0:
        #     weights = self.model.get_weights()
        #     for w in weights:
        #         print(w)

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
        processed_new_state = self.__process_state(new_state)
        if prev_state is not None:
            processed_prev_state = self.__process_state(prev_state)
            self.memory.append([processed_prev_state, prev_action, reward, processed_new_state])
        if np.random.random() < self.epsilon or prev_state is None:
            return r.sample(self.ACTIONS, 1)[0]
        return self.ACTIONS[np.argmax(self.model.predict(processed_new_state))]

    def __process_state_1(self, state):
        """
        Extracts the part of the board which is within RADIUS around the snake.
        :param state: A tuple (board, head) representing the state.
        :return: A numpy array representing the window.
        """
        board, head_pos = state[0], state[1][0]
        window = np.zeros((WINDOW_SIDE_LENGTH, WINDOW_SIDE_LENGTH))
        for i in range(-RADIUS, RADIUS + 1):
            for j in range(-RADIUS, RADIUS + 1):
                window[i + RADIUS, j + RADIUS] = board[(head_pos[0] + i + self.board_size[0]) % self.board_size[0], (
                            head_pos[1] + j + self.board_size[1]) % self.board_size[1]]
        return window.reshape((1, NUM_ELEMENTS))

    def __process_state_2(self, state):
        """
        Extracts the part of the board which is within RADIUS around the snake.
        :param state: A tuple (board, head) representing the state.
        :return: A numpy array representing the window.
        """
        board, head_pos = state[0], state[1][0]
        window = np.zeros((WINDOW_SIDE_LENGTH, WINDOW_SIDE_LENGTH))
        for i in range(-RADIUS, RADIUS + 1):
            for j in range(-RADIUS, RADIUS + 1):
                window[i + RADIUS, j + RADIUS] = board[(head_pos[0] + i + self.board_size[0]) % self.board_size[0], (
                            head_pos[1] + j + self.board_size[1]) % self.board_size[1]]
        return window.reshape((1, WINDOW_SIDE_LENGTH, WINDOW_SIDE_LENGTH, 1))

    def __process_state_3(self, state):
        """
        Extracts the part of the board which is within RADIUS around the snake.
        :param state: A tuple (board, head) representing the state.
        :return: A numpy array representing the window.
        """
        board, head_pos = state[0], state[1][0]
        window = np.zeros((WINDOW_SIDE_LENGTH, WINDOW_SIDE_LENGTH))
        for i in range(-RADIUS, RADIUS + 1):
            for j in range(-RADIUS, RADIUS + 1):
                window[i + RADIUS, j + RADIUS] = board[(head_pos[0] + i + self.board_size[0]) % self.board_size[0], (
                            head_pos[1] + j + self.board_size[1]) % self.board_size[1]]
        return to_categorical(window, num_classes=VALUES).reshape((1, NUM_ELEMENTS * VALUES))

    def __process_state_4(self, state):
        board, head_pos = state[0], state[1][0]
        rows = [i % self.board_size[0] for i in
                range(head_pos[0] - RADIUS + self.board_size[0], head_pos[0] + RADIUS + self.board_size[0] + 1)]
        cols = [i % self.board_size[1] for i in
                range(head_pos[1] - RADIUS + self.board_size[1], head_pos[1] + RADIUS + self.board_size[1] + 1)]
        window = board[rows, :][:, cols]
        # return window[np.newaxis, ..., np.newaxis]
        # return to_categorical(window, num_classes=VALUES).reshape((1, NUM_ELEMENTS * VALUES))
        return window.reshape((1, NUM_ELEMENTS))

    def __process_state_5(self, state):
        board, head_pos = state[0], state[1][0]
        rows = [i % self.board_size[0] for i in
                range(head_pos[0] - RADIUS + self.board_size[0], head_pos[0] + RADIUS + self.board_size[0] + 1)]
        cols = [i % self.board_size[1] for i in
                range(head_pos[1] - RADIUS + self.board_size[1], head_pos[1] + RADIUS + self.board_size[1] + 1)]
        window = board[rows, :][:, cols]
        # return window[np.newaxis, ..., np.newaxis]
        return to_categorical(window, num_classes=VALUES).reshape((1, NUM_ELEMENTS * VALUES))
        # return window.reshape((1, NUM_ELEMENTS))

    def create_model_1(self):
        model = Sequential()
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        # model.add(Dense(64, activation='tanh'))
        model.add(Dense(64, activation='tanh', kernel_regularizer=l2()))
        model.add(Dense(len(self.ACTIONS)))
        model.compile(optimizer=Adam(lr=self.lr), loss='mean_squared_error')
        return model

    def create_model_2(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=5, activation='tanh'))
        model.add(Conv2D(filters=64, kernel_size=3, activation='tanh'))
        model.add(Flatten())
        model.add(Dense(256, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(len(self.ACTIONS)))
        model.compile(optimizer=Adam(lr=self.lr), loss='mean_squared_error')
        return model

    def create_model_3(self):
        model = Sequential()
        model.add(Dense(128, activation='tanh', kernel_regularizer=l2()))
        model.add(Dense(len(self.ACTIONS), kernel_regularizer=l2()))
        model.compile(optimizer=Adam(lr=self.lr), loss='mean_squared_error')
        return model

    def create_model_4(self):
        model = Sequential()
        # model.add(Conv2D(32, 5, activation='relu'))
        # model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(len(self.ACTIONS)))
        model.compile(optimizer=Adam(lr=self.lr), loss='mean_squared_error')
        return model
