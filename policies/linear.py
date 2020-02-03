import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
from policies.base_policy import Policy
from collections import deque
import random as rnd

LEARNING_RATE = 1e-3
EPSILON = 1.0
BATCH_SIZE = 25
BATCH_THRESHOLD = 300
NUM_FEATURES = 44  # there are 4 adjacent positions to the head of the snake, each holding one of 11 possible values
VALUES = 11
GAMMA = 0.95


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
        self.weights = np.random.random(NUM_FEATURES).astype('float32')
        self.r_sum = 0
        self.replay_buffer = deque(maxlen=1000)
        self.window = 3  # should check different sizes, this is just an initial value
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate_min = 0.1
        self.learning_rate_decay = 0.995
        # self.model = self.get_model()
        self.batch_size = BATCH_SIZE
        self.epsilon = self.__dict__['epsilon']
        self.learning_rate = self.__dict__['lr']
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
        if prev_state is None:
            # This will only happen at the beginning of the game, in which case we cannot actually learn.
            return
        # update the weights, epsilon and lr
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.learning_rate > self.learning_rate_min:
            self.learning_rate *= self.learning_rate_decay

        # train on batch
        if len(self.replay_buffer) > BATCH_THRESHOLD:
            minibatch = rnd.sample(self.replay_buffer, self.batch_size)
            for processed_prev, prev_action, reward, processed_new, new_action in minibatch:
                # if the action we get now is different from the action we picked - retrain
                predictions = self.get_q_values(processed_new, self.__get_global_direction(processed_prev, processed_new))
                if self.ACTIONS[np.argmax(predictions)] != new_action:
                    self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * (
                                reward + self.gamma * np.max(
                            self.get_q_values(processed_new, self.__get_global_direction(processed_prev, processed_new))))

        # train on current
        self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * \
                       (reward + self.gamma *
                        np.max(self.get_q_values(new_state, self.__get_global_direction(prev_state, new_state))))

        # TODO - If the linear model works, remove the code below
        # if not len(self.replay_buffer):
        #     return

        # make batch size smaller to decrease computation time
        # if too_slow or self.batch_size > round:
        #     self.batch_size //= 2

        # minibatch = np.random.choice(self.replay_buffer, self.batch_size)
        # for processed_prev, prev_action, reward, processed_new, new_action in minibatch:
        #     target = reward + self.gamma * \
        #              np.amax(self.model.predict(processed_new)[0])
        #     target_f = self.model.predict(processed_prev)
        #     target_f[0][new_action] = target
        #     self.model.fit(processed_prev, target_f, epochs=1, verbose=0)
        #
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        #
        # try:
        #     if round % 100 == 0:
        #         if round > self.game_duration - self.score_scope:
        #             self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
        #         else:
        #             self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
        #         self.r_sum = 0
        #     else:
        #         self.r_sum += reward
        #
        # except Exception as e:
        #     self.log("Something Went Wrong...", 'EXCEPTION')
        #     self.log(e, 'EXCEPTION')

    def __get_global_direction(self, prev_state, current_state):
        prev_head = prev_state[1]
        prev_head_pos = prev_head[0]
        curr_head = current_state[1]
        curr_head_pos = curr_head[0]
        if prev_head_pos[0] == curr_head_pos[0]:
            # moving horizontally
            if prev_head_pos[1] < curr_head_pos[1]:
                # moving to the east
                return 'E'
            else:
                # moving to the west
                return 'W'
        else:
            # moving vertically
            if prev_head_pos[0] < curr_head_pos[0]:
                # moving to the north
                return 'N'
            else:
                # moving to the south
                return 'S'

    def __get_next_head_pos(self, state, action, global_direction):
        """"""
        head = state[1]
        if global_direction == 'N':
            l, r, f = (0, -1), (0, 1), (-1, 0)
        elif global_direction == 'S':
            l, r, f = (0, 1), (0, -1), (1, 0)
        elif global_direction == 'E':
            l, r, f = (-1, 0), (1, 0), (0, 1)
        else:
            l, r, f = (1, 0), (-1, 0), (0, -1)
        if action == "L":
            return head + l
        elif action == "R":
            return head + r
        return head + f

    def __feature_function(self, state, next_head_pos):
        """
        Parses the given state into pre-determined features.
        The function encodes the values in the 4 adjacent positions to the snake's head into an indicator vector.
        :param state: A state as defined in the exercise requirements.
        :return: An array of features representing the state.
        """
        features = np.zeros(NUM_FEATURES)
        board, head = state
        pos = next_head_pos[0]
        features[1 + board[(pos[0] - 1) % self.board_size[0], pos[1]]] = 1  # encoding of the value in the position above the snake's head
        features[VALUES + 1 + board[pos[0], (pos[1] - 1) % self.board_size[1]]] = 1  # encoding of the value in the position to the left of the snake's head
        features[2 * VALUES + 1 + board[(pos[0] + 1) % self.board_size[0], pos[1]]] = 1  # encoding of the value in the position below the snake's head
        features[3 * VALUES + 1 + board[pos[0], (pos[1] + 1) % self.board_size[1]]] = 1  # encoding of the value in the position to the right of the snake's head
        return features

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
        if np.random.rand() < self.epsilon or prev_state is None:
            action = np.random.choice(Policy.ACTIONS)
            if prev_state is not None:
                self.replay_buffer.append((prev_state, prev_action, reward, new_state, action))
            return action
        action = self.ACTIONS[np.argmax(self.get_q_values(new_state, self.__get_global_direction(prev_state, new_state)))]
        self.replay_buffer.append((prev_state, prev_action, reward, new_state, action))
        return action

        # processed_new = self.process_state(new_state)
        # choice = np.random.choice(3, 1, p=self.weights)[0]
        # new_action = Policy.ACTIONS[choice]
        # if prev_state is not None:
        #     processed_prev = self.process_state(prev_state)
        #     self.replay_buffer.append((processed_prev, prev_action, reward, processed_new, new_action))
        # return new_action

    def get_q_values(self, state, global_direction):
        q_vals = np.zeros(len(Policy.ACTIONS))
        for i, action in enumerate(Policy.ACTIONS):
            next_head_pos = self.__get_next_head_pos(state, action, global_direction)
            q_vals[i] = np.dot(self.weights, self.__feature_function(state, next_head_pos))
        return q_vals

    def process_state(self, state):
        """
        Reduces state dimension.
        :param state: tuple (board, head) representing the state to process.
        :return: A tuple (precessed, head) representing a processed state.
        """
        board, head = state
        tl = (max(0, (head[0] - self.window) % 10), max(0, (head[1] - self.window) % 10))
        # tr = (max(0, (head[0] - self.window) % 10), min(10, (head[1] + self.window + 1) % 10))  # might use later...
        # bl = (max(0, (head[0] + self.window + 1) % 10), min(10, (head[1] - self.window) % 10))  # might use later...
        br = (max(0, (head[0] + self.window + 1) % 10), max(0, (head[0] + self.window + 1) % 10))
        vertical = tl[0] < br[0]
        horizontal = tl[1] < br[1]
        if vertical and horizontal:
            window = board[tl[0]: br[0], tl[1]: br[1]]
        elif vertical and not horizontal:
            window = np.concatenate((board[tl[0]: br[0], tl[1]: self.board_size[1]], board[tl[0]: br[0], : br[1]]),
                                    axis=1)
        elif horizontal and not vertical:
            window = np.concatenate((board[tl: self.board_size[0], tl[1]: br[1]], board[: br[0], tl[1]: br[1]]))
        else:
            window = np.concatenate((np.concatenate((board[tl[0]: self.board_size[0], tl[1]: self.board_size[1]],
                                                     board[tl[0]: self.board_size[0], 0: br[1]]), axis=1),
                                     np.concatenate((board[: br[0], tl[1]: self.board_size[1]],
                                                     board[: br[0], : br[1]]), axis=1)))
        return window, head

    # def get_model(self):
    #     return some tensorflow/keras model with the functions predict and fit
        # model = Sequential()
        # model.add(Dense(3, activation='linear'))
        # return model
