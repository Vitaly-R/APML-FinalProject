from policies.base_policy import Policy
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random as r


INITIAL_EPSILON = 1
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
LEARNING_RATE = 1e-2
RADIUS = 5
WINDOW_SIDE_LENGTH = 2 * RADIUS + 1
NUM_ELEMENTS = WINDOW_SIDE_LENGTH ** 2
GAMMA = 0.95
BATCH_SIZE = 32


class CustomPolicy(Policy):
    def cast_string_args(self, policy_args):
        """
        this function casts arguments passed during policy construction to their proper types/names.
        :param policy_args: an arg -> string value map as received in command line.
        :return: A map of string -> value after casting to useful objects, these will be added as members to the policy
        """
        return policy_args

    def init_run(self):
        """
        this function is called right after the initialization of the agent.
        you may use it to initialize variables that are needed for your policy,
        such as Keras models and so on. you may also use this function
        to load your pickled model and set the variables accordingly, if the
        game uses a saved model and is not a training session.
        """
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.model.predict(np.zeros((1, WINDOW_SIDE_LENGTH, WINDOW_SIDE_LENGTH, 1)))
        self.epsilon = INITIAL_EPSILON
        self.memory = deque(maxlen=2000)
        self.t = 0.125

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
        if len(self.memory) < BATCH_SIZE:
            return
        samples = r.sample(self.memory, BATCH_SIZE)
        for sample in samples:
            state, action, reward, new_state = sample
            target = self.target_model.predict(state)
            q_future = max(self.target_model.predict(self.__process_state(new_state))[0])
            target[0][self.ACTIONS.index(action)] = reward + GAMMA * q_future
            self.model.fit(state, target, epochs=1, verbose=0)
        
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
        processed_prev_state = self.__process_state(prev_state)
        processed_new_state = self.__process_state(new_state)
        if prev_state is not None:
            self.memory.append([processed_prev_state, prev_action, reward, processed_new_state])
        if np.random.random() < self.epsilon or prev_state is None:
            return r.sample(self.ACTIONS, 1)[0]
        return self.ACTIONS[np.argmax(self.model.predict(processed_new_state))]

    def __process_state(self, state):
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

    def create_model(self):
        model = Sequential()
        model.add(Dense(512, activation='tanh'))
        model.add(Dense(len(self.ACTIONS)))
        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mean_squared_error')
        return model
