import numpy as np
from policies.base_policy import Policy
from collections import deque
import random as rnd

VALUES = 11
LEARNING_RATE = 1e-2
EPSILON = 1.0
BATCH_SIZE = 10
MAX_BATCH_SIZE = 20
MIN_BATCH_SIZE = 3
BATCH_THRESHOLD = 300
RADIUS = 2
WINDOW_SIDE_LENGTH = (2 * RADIUS + 1)
NUM_FEATURES = (WINDOW_SIDE_LENGTH ** 2) * VALUES  # 11 possible values for each of the elements in the window
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
        self.weights = np.random.random(NUM_FEATURES).astype('float32')
        self.r_sum = 0
        self.replay_buffer = deque(maxlen=BATCH_THRESHOLD + 100)
        self.window = 3  # should check different sizes, this is just an initial value
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.995
        self.learning_rate_min = 1e-4
        self.learning_rate_decay = 0.99
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
        if round > BATCH_THRESHOLD:
            if too_slow:
                self.batch_size = max(self.batch_size // 2, MIN_BATCH_SIZE)
            elif not round % 100:  # to prevent from happening too often
                self.batch_size = min(self.batch_size + 1, MAX_BATCH_SIZE)
            minibatch = rnd.sample(self.replay_buffer, self.batch_size)
            for prev_s, prev_action, reward, new_s, new_action in minibatch:
                global_direction = self.__get_global_direction(prev_s, new_s)
                max_q_val = np.nanmax(self.get_q_values(new_s, global_direction))
                self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * (reward +
                                                                                               self.gamma * max_q_val) * self.__process_state(prev_s, prev_s[1][0])

        # train on current
        global_direction = self.__get_global_direction(prev_state, new_state)
        max_q_val = np.nanmax(self.get_q_values(new_state, global_direction))
        self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * (reward +
                                                                                       self.gamma * max_q_val) * self.__process_state(prev_state, prev_state[1][0])
        # normalize weights
        if round > 2 * BATCH_THRESHOLD:
            norm = np.linalg.norm(self.weights)
            if norm:
                self.weights /= norm

        #
        # if not round % 500:  # TODO: delete this
        #     # print weights
        #     # print(self.weights)
        #     print("max weight: " + str(np.max(self.weights)))
        #     print("min weight: " + str(np.min(self.weights)))
        #     print("positives amount: " + str(len(self.weights[self.weights > 0])))
        #     print("negatives amount: " + str(len(self.weights[self.weights <= 0])))

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
        head_pos = state[1][0]
        if global_direction == 'N':
            l, r, f = (0, -1), (0, 1), (-1, 0)
        elif global_direction == 'S':
            l, r, f = (0, 1), (0, -1), (1, 0)
        elif global_direction == 'E':
            l, r, f = (-1, 0), (1, 0), (0, 1)
        else:
            l, r, f = (1, 0), (-1, 0), (0, -1)
        if action == "L":
            return head_pos + l
        elif action == "R":
            return head_pos + r
        return head_pos + f

    # @staticmethod
    def __process_state(self, state, head_pos):
        """
        Parses the given state into pre-determined features.
        The function encodes the values in the 4 adjacent positions to the snake's head into an indicator vector.
        :param state: A state as defined in the exercise requirements.
        :return: An array of features representing the state.
        """
        features = np.zeros(NUM_FEATURES)
        board = state[0]
        for i in range(-RADIUS, RADIUS + 1):
            for j in range(-RADIUS, RADIUS + 1):
                ridx = ((head_pos[0] + i + self.board_size[0]) % self.board_size[0])
                cidx = (((head_pos[1] + j) + self.board_size[1]) % self.board_size[1])
                value = board[ridx, cidx] + 1  # +1 to shift it to range 0-10
                features[((i + RADIUS) * WINDOW_SIDE_LENGTH + (j + RADIUS)) * WINDOW_SIDE_LENGTH + int(value)] = 1
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

    def get_q_values(self, state, global_direction):
        q_vals = np.zeros(len(Policy.ACTIONS))
        for i, action in enumerate(Policy.ACTIONS):
            next_head_pos = self.__get_next_head_pos(state, action, global_direction)
            next_state = self.get_next_state(state, action, global_direction)
            q_vals[i] = np.dot(self.weights, self.__process_state(next_state, next_head_pos))
        return q_vals

    def get_next_state(self, state, action, global_direction):
        '''
        TODO: find how to get the next state with the current state and direction, if possible
        :param state:
        :param action:
        :param global_direction:
        :return:
        '''
        return state

# if __name__ == '__main__':
#     state = np.zeros((20, 60))
#     for i in range(20):
#         state[i, :] = i % 11 - 1
#     head = Snake.Position([19, 59], [20, 60])
#     features = LinearAgent.process_state((state, [head]), [20, 60])
#     print(features.shape)
#     print(np.argwhere(features == 1))

