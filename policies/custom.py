from policies.base_policy import Policy
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from collections import deque
import numpy as np
import random as r


# todo: add the following to the answers file:
# The weights of the target model need to be updated because the main model learns from playing the game in relation to the target model,
# but we also need to update the target model to improve it based on the weights of the main model.


VALUES = 11
LEARNING_RATE = 1e-3
BATCH_SIZE = 10
MAX_BATCH_SIZE = 20
MIN_BATCH_SIZE = 5
BATCH_THRESHOLD = 500
RADIUS = 7
GAMMA = 0.85
INITIAL_EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.05
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
        else:
            print('creating convolutional network')
            self.model = self.create_model_3()
            self.target_model = self.create_model_3()
            self.model.predict(np.zeros((1, NUM_ELEMENTS)))
            self.__process_state = self.__process_state_1
        self.memory = deque(maxlen=2*BATCH_THRESHOLD)
        self.batch_size = BATCH_SIZE
        self.t = 0.125  # a learning rate for the weights of the target model in relation to the main model
        self.epsilon = self.__dict__['epsilon']
        self.lr = self.__dict__['lr']
        self.gamma = self.__dict__['gamma']

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
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
        if round > BATCH_THRESHOLD:
            # In order for the model not to start training too early, we wait until enough states are saved.
            if too_slow:
                self.batch_size = max(self.batch_size // 2, MIN_BATCH_SIZE)
            elif not round % 100:  # to prevent from happening too often
                self.batch_size = min(self.batch_size + 1, MAX_BATCH_SIZE)
            samples = r.sample(self.memory, self.batch_size)
            for sample in samples:
                state, action, reward, next_state = sample
                target = self.target_model.predict(state)  # q-values of the target model for each action
                q_future = max(self.target_model.predict(next_state)[0])  # the max q-value possible from the next state
                target[0][self.ACTIONS.index(action)] = reward + self.gamma * q_future  # the target q-value we want our model to achieve for the specific action
                self.model.fit(state, target, epochs=1, verbose=0)  # training and updating the weights of the main model for the target activation.

            # updating the weights of the target model in relation to the main model.
            weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = weights[i] * self.t + target_weights[i] * (1 - self.t)
            self.target_model.set_weights(target_weights)
        # # # TODO: delete this  -- DOESNT WORK FOR NOW, list of arrays?
        # if round > BATCH_THRESHOLD and not round % 500:
        #     # print weights
        #     # print(self.weights)
        #     weights = self.model.get_weights()
        #     for w_l in weights:
        #         print("max weight: " + str(np.max(w_l)))
        #         print("min weight: " + str(np.min(w_l)))
        #         print("positives amount: " + str(len(w_l[w_l > 0])))
        #         print("negatives amount: " + str(len(w_l[w_l <= 0])))
        #         print("**********************")

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
                window[i + RADIUS, j + RADIUS] = board[(head_pos[0] + i + self.board_size[0]) % self.board_size[0], (head_pos[1] + j + self.board_size[1]) % self.board_size[1]]
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
                window[i + RADIUS, j + RADIUS] = board[(head_pos[0] + i + self.board_size[0]) % self.board_size[0], (head_pos[1] + j + self.board_size[1]) % self.board_size[1]]
        return window.reshape((1, WINDOW_SIDE_LENGTH, WINDOW_SIDE_LENGTH, 1))

    def create_model_1(self):
        model = Sequential()
        model.add(Dense(128, activation='tanh', input_shape=(NUM_ELEMENTS, )))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(64, activation='tanh'))
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
        model.add(Dense(256, activation='tanh', kernel_regularizer=l2()))
        model.add(Dense(len(self.ACTIONS)))
        model.compile(optimizer=Adam(lr=self.lr), loss='mean_squared_error')
        return model
