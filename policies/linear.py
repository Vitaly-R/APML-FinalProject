import numpy as np
from policies.base_policy import Policy
from collections import deque

LEARNING_RATE = 1e-3
EPSILON = 0.05
BATCH_SIZE = 2


class LinearAgent(Policy):

    def cast_string_args(self, policy_args):
        """
        this function casts arguments passed during policy construction to their proper types/names.
        :param policy_args: an arg -> string value map as received in command line.
        :return: A map of string -> value after casting to useful objects, these will be added as members to the policy
        """
        policy_args['lr'] = float(policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def init_run(self):
        """
        this function is called right after the initialization of the agent.
        you may use it to initialize variables that are needed for your policy,
        such as Keras models and so on. you may also use this function
        to load your pickled model and set the variables accordingly, if the
        game uses a saved model and is not a training session.
        """
        self.weights = np.ones(3).astype('float32') / 3.0
        self.r_sum = 0
        self.replay_buffer = deque(maxlen=1000)
        self.window = 3  # should check different sizes, this is just an initial value
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.get_model()

    def adjust_weights(self, new_rewards):
        pass

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
        if not len(self.replay_buffer):
            return

        # make batch size smaller to decrease computation time
        if too_slow or BATCH_SIZE > round:
            BATCH_SIZE //= 2

        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        for processed_prev, prev_action, reward, processed_new, new_action in minibatch:
            target = reward + self.gamma * \
                     np.amax(self.model.predict(processed_new)[0])
            target_f = self.model.predict(processed_prev)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

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
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        processed_new = self.process_state(new_state)
        choice = np.random.choice(3, 1, p=self.weights)[0]
        new_action = Policy.ACTIONS[choice]
        if prev_state is not None:
            processed_prev = self.process_state(prev_state)
            self.replay_buffer.append((processed_prev, prev_action, reward, processed_new, new_action))
        return new_action

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

    def get_model(self):
        # return some tensorflow/keras model with the functions predict and fit
        return
