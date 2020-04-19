# Do not modify this cell!

# Import necessary libraries
# DO NOT IMPORT OTHER LIBRARIES - This will break the autograder.
import numpy as np
import matplotlib.pyplot as plt

import os
import shutil
from tqdm import tqdm

from rl_glue import RLGlue
from environment import BaseEnvironment
from agent import BaseAgent
from optimizer import BaseOptimizer
import plot_script
from randomwalk_environment import RandomWalkEnvironment


def my_matmul(x1, x2):
    """
    Given matrices x1 and x2, return the multiplication of them
    """

    result = np.zeros((x1.shape[0], x2.shape[1]))
    x1_non_zero_indices = x1.nonzero()
    if x1.shape[0] == 1 and len(x1_non_zero_indices[1]) == 1:
        result = x2[x1_non_zero_indices[1], :]
    elif x1.shape[1] == 1 and len(x1_non_zero_indices[0]) == 1:
        result[x1_non_zero_indices[0], :] = x2 * x1[x1_non_zero_indices[0], 0]
    else:
        result = np.matmul(x1, x2)
    return result


# GRADED FUNCTION: [get_value]
def get_value(s, weights):
    """
    Compute value of input s given the weights of a neural network
    """
    # Compute the ouput of the neural network, v, for input s (3 lines)
    ### START CODE HERE ###
    o = s
    for i, w in enumerate(weights):
        p = np.add(my_matmul(o, w["W"]), w["b"])
        if i < len(weights) - 1:
            o = np.maximum(p, np.zeros(p.shape))
    v = p
    ### END CODE HERE ###
    return v


# GRADED FUNCTION: [get_gradient]
def get_gradient(s, weights):
    """
    Given inputs s and weights, return the gradient of v with respect to the weights
    """

    # Compute the gradient of the value function with respect to W0, b0, W1, b1 for input s (6~8 lines)
    # grads[0]["W"] = ?
    # grads[0]["b"] = ?
    # grads[1]["W"] = ?
    # grads[1]["b"] = ?
    # Note that grads[0]["W"], grads[0]["b"], grads[1]["W"], and grads[1]["b"] should have the same shape as
    # weights[0]["W"], weights[0]["b"], weights[1]["W"], and weights[1]["b"] respectively
    # Note that to compute the gradients, you need to compute the activation of the hidden layer (x)

    grads = [dict() for i in range(len(weights))]

    ### START CODE HERE ###
    p = np.add(my_matmul(s, weights[0]["W"]), weights[0]["b"])
    x = np.maximum(p, np.zeros(p.shape))
    I = np.array([1 if v > 0 else 1 for v in x[0]])
    grads[0]["b"] = np.multiply(np.transpose(weights[1]["W"]), I)
    grads[0]["W"] = my_matmul(np.transpose(s), grads[0]["b"])
    grads[1]["b"] = 1.0
    grads[1]["W"] = np.transpose(x)
    ### END CODE HERE ###

    return grads


class Adam(BaseOptimizer):
    def __init__(self):
        pass

    def optimizer_init(self, optimizer_info):
        """Setup for the optimizer.

        Set parameters needed to setup the Adam algorithm.

        Assume optimizer_info dict contains:
        {
            num_states: integer,
            num_hidden_layer: integer,
            num_hidden_units: integer,
            step_size: float, 
            self.beta_m: float
            self.beta_v: float
            self.epsilon: float
        }
        """

        self.num_states = optimizer_info.get("num_states")
        self.num_hidden_layer = optimizer_info.get("num_hidden_layer")
        self.num_hidden_units = optimizer_info.get("num_hidden_units")

        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")

        self.layer_size = np.array([self.num_states, self.num_hidden_units, 1])

        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(self.num_hidden_layer+1)]
        self.v = [dict() for i in range(self.num_hidden_layer+1)]

        for i in range(self.num_hidden_layer+1):

            # Initialize self.m[i]["W"], self.m[i]["b"], self.v[i]["W"], self.v[i]["b"] to zero
            self.m[i]["W"] = np.zeros(
                (self.layer_size[i], self.layer_size[i+1]))
            self.m[i]["b"] = np.zeros((1, self.layer_size[i+1]))
            self.v[i]["W"] = np.zeros(
                (self.layer_size[i], self.layer_size[i+1]))
            self.v[i]["b"] = np.zeros((1, self.layer_size[i+1]))

        # Initialize beta_m_product and beta_v_product to be later used for computing m_hat and v_hat
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

    def update_weights(self, weights, g):
        """
        Given weights and update g, return updated weights
        """

        for i in range(len(weights)):
            for param in weights[i].keys():

                # update self.m and self.v
                self.m[i][param] = self.beta_m * self.m[i][param] + \
                    (1 - self.beta_m) * g[i][param]
                self.v[i][param] = self.beta_v * self.v[i][param] + \
                    (1 - self.beta_v) * (g[i][param] * g[i][param])

                # compute m_hat and v_hat
                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)

                # update weights
                weights[i][param] += self.step_size * \
                    m_hat / (np.sqrt(v_hat) + self.epsilon)

        # update self.beta_m_product and self.beta_v_product
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        return weights


def one_hot(state, num_states):
    """
    Given num_state and a state, return the one-hot encoding of the state
    """
    # Create the one-hot encoding of state
    # one_hot_vector is a numpy array of shape (1, num_states)

    one_hot_vector = np.zeros((1, num_states))
    one_hot_vector[0, int((state - 1))] = 1

    return one_hot_vector

# GRADED FUNCTION: [Agent]


class TDAgent(BaseAgent):
    def __init__(self):
        self.name = "td_agent"
        pass

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD with a Neural Network.

        Assume agent_info dict contains:
        {
            num_states: integer,
            num_hidden_layer: integer,
            num_hidden_units: integer,
            step_size: float, 
            discount_factor: float,
            self.beta_m: float
            self.beta_v: float
            self.epsilon: float
            seed: int
        }
        """

        # Set random seed for weights initialization for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        # Set random seed for policy for each run
        self.policy_rand_generator = np.random.RandomState(
            agent_info.get("seed"))

        # Set attributes according to agent_info
        self.num_states = agent_info.get("num_states")
        self.num_hidden_layer = agent_info.get("num_hidden_layer")
        self.num_hidden_units = agent_info.get("num_hidden_units")
        self.discount_factor = agent_info.get("discount_factor")

        # Define the neural network's structure (1 line)
        # Specify self.layer_size which shows the number of nodes in each layer
        # self.layer_size = np.array([None, None, None])
        # Hint: Checkout the NN diagram at the beginning of the notebook

        ### START CODE HERE ###
        self.layer_size = np.array([self.num_states, self.num_hidden_units, 1])
        ### END CODE HERE ###

        # Initialize the neural network's parameter (2 lines)
        self.weights = [dict() for i in range(self.num_hidden_layer+1)]
        for i in range(self.num_hidden_layer+1):

            # Initialize self.weights[i]["W"] and self.weights[i]["b"] using self.rand_generator.normal()
            # Note that The parameters of self.rand_generator.normal are mean of the distribution,
            # standard deviation of the distribution, and output shape in the form of tuple of integers.
            # To specify output shape, use self.layer_size.

            ### START CODE HERE ###
            self.weights[i]["W"] = self.rand_generator.normal(0, np.sqrt(
                2 / self.layer_size[i]), (self.layer_size[i], self.layer_size[i + 1]))
            self.weights[i]["b"] = self.rand_generator.normal(
                0, np.sqrt(2 / self.layer_size[i]), (1, self.layer_size[i + 1]))
            ### END CODE HERE ###

        # Specify the optimizer
        self.optimizer = Adam()
        optimizer_info = {"num_states": agent_info["num_states"],
                          "num_hidden_layer": agent_info["num_hidden_layer"],
                          "num_hidden_units": agent_info["num_hidden_units"],
                          "step_size": agent_info["step_size"],
                          "beta_m": agent_info["beta_m"],
                          "beta_v": agent_info["beta_v"],
                          "epsilon": agent_info["epsilon"]}
        self.optimizer.optimizer_init(optimizer_info)

        self.last_state = None
        self.last_action = None

    def agent_policy(self, state):

        # Set chosen_action as 0 or 1 with equal probability.
        chosen_action = self.policy_rand_generator.choice([0, 1])
        return chosen_action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        # select action given state (using self.agent_policy()), and save current state and action (2 lines)
        # self.last_state = ?
        # self.last_action = ?

        ### START CODE HERE ###
        self.last_state = state
        self.last_action = self.agent_policy(state)
        ### END CODE HERE ###

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

        # Compute TD error (5 lines)
        # delta = None

        ### START CODE HERE ###
        s_current = one_hot(state, self.num_states)
        s_last = one_hot(self.last_state, self.num_states)

        v_h_current = get_value(s_current, self.weights)
        v_h_last = get_value(s_last, self.weights)
        print(reward, self.discount_factor, v_h_current, v_h_last)

        delta = reward + self.discount_factor * v_h_current - v_h_last
        print("delta", delta, delta.shape)
        ### END CODE HERE ###

        # Retrieve gradients (1 line)
        # grads = None

        ### START CODE HERE ###
        grads = get_gradient(s_last, self.weights)
        ### END CODE HERE ###

        print("grads", grads)

        # Compute g (1 line)
        g = [dict() for i in range(self.num_hidden_layer+1)]
        for i in range(self.num_hidden_layer+1):
            for param in self.weights[i].keys():

                # g[i][param] = None
                ### START CODE HERE ###
                g[i][param] = delta * grads[i][param]
                ### END CODE HERE ###

        print("g", g)

        # update the weights using self.optimizer (1 line)
        # self.weights = None

        ### START CODE HERE ###
        self.weights = self.optimizer.update_weights(self.weights, g)
        ### END CODE HERE ###

        # update self.last_state and self.last_action (2 lines)

        ### START CODE HERE ###
        self.last_state = state
        self.last_action = self.agent_policy(state)
        ### END CODE HERE ###

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # compute TD error (3 lines)
        # delta = None

        ### START CODE HERE ###
        s_last = one_hot(self.last_state, self.num_states)
        delta = reward - get_value(s_last, self.weights)
        ### END CODE HERE ###

        # Retrieve gradients (1 line)
        # grads = None

        ### START CODE HERE ###
        grads = get_gradient(s_last, self.weights)
        ### END CODE HERE ###

        # Compute g (1 line)
        g = [dict() for i in range(self.num_hidden_layer+1)]
        for i in range(self.num_hidden_layer+1):
            for param in self.weights[i].keys():

                # g[i][param] = None
                ### START CODE HERE ###
                g[i][param] = delta * grads[i][param]
                ### END CODE HERE ###

        # update the weights using self.optimizer (1 line)
        # self.weights = None

        ### START CODE HERE ###
        self.weights = self.optimizer.update_weights(self.weights, g)
        ### END CODE HERE ###

    def agent_message(self, message):
        if message == 'get state value':
            state_value = np.zeros(self.num_states)
            for state in range(1, self.num_states + 1):
                s = one_hot(state, self.num_states)
                state_value[state - 1] = get_value(s, self.weights)
            return state_value

## Test Code for agent_init() ##


def test_get_value():
    ## Test Code for get_value() ##

    # Suppose num_states = 5, num_hidden_layer = 1, and num_hidden_units = 10
    num_hidden_layer = 1
    s = np.array([[0, 0, 0, 1, 0]])

    weights_data = np.load("asserts/get_value_weights.npz")
    weights = [dict() for i in range(num_hidden_layer+1)]
    weights[0]["W"] = weights_data["W0"]
    weights[0]["b"] = weights_data["b0"]
    weights[1]["W"] = weights_data["W1"]
    weights[1]["b"] = weights_data["b1"]

    estimated_value = get_value(s, weights)
    print("Estimated value: {}".format(estimated_value))
    assert(np.allclose(estimated_value, np.array([[-0.21915705]])))

    print("Passed the assert!")


def test_get_gradient():
    ## Test Code for get_gradient() ##

    # Suppose num_states = 5, num_hidden_layer = 1, and num_hidden_units = 2
    num_hidden_layer = 1
    s = np.array([[0, 0, 0, 1, 0]])

    weights_data = np.load("asserts/get_gradient_weights.npz")
    weights = [dict() for i in range(num_hidden_layer+1)]
    weights[0]["W"] = weights_data["W0"]
    weights[0]["b"] = weights_data["b0"]
    weights[1]["W"] = weights_data["W1"]
    weights[1]["b"] = weights_data["b1"]

    grads = get_gradient(s, weights)

    grads_answer = np.load("asserts/get_gradient_grads.npz")

    print("grads[0][\"W\"]\n", grads[0]["W"], "\n")
    print("grads[0][\"b\"]\n", grads[0]["b"], "\n")
    print("grads[1][\"W\"]\n", grads[1]["W"], "\n")
    print("grads[1][\"b\"]\n", grads[1]["b"], "\n")

    assert(np.allclose(grads[0]["W"], grads_answer["W0"]))
    assert(np.allclose(grads[0]["b"], grads_answer["b0"]))
    assert(np.allclose(grads[1]["W"], grads_answer["W1"]))
    assert(np.allclose(grads[1]["b"], grads_answer["b1"]))

    print("Passed the asserts!")


def test_agent_init():
    agent_info = {"num_states": 5,
                  "num_hidden_layer": 1,
                  "num_hidden_units": 2,
                  "step_size": 0.25,
                  "discount_factor": 0.9,
                  "beta_m": 0.9,
                  "beta_v": 0.99,
                  "epsilon": 0.0001,
                  "seed": 0
                  }

    test_agent = TDAgent()
    test_agent.agent_init(agent_info)

    print("layer_size: {}".format(test_agent.layer_size))
    assert(np.allclose(test_agent.layer_size, np.array([agent_info["num_states"],
                                                        agent_info["num_hidden_units"],
                                                        1])))

    print("weights[0][\"W\"] shape: {}".format(
        test_agent.weights[0]["W"].shape))
    print("weights[0][\"b\"] shape: {}".format(
        test_agent.weights[0]["b"].shape))
    print("weights[1][\"W\"] shape: {}".format(
        test_agent.weights[1]["W"].shape))
    print("weights[1][\"b\"] shape: {}".format(
        test_agent.weights[1]["b"].shape), "\n")

    assert(test_agent.weights[0]["W"].shape == (
        agent_info["num_states"], agent_info["num_hidden_units"]))
    assert(test_agent.weights[0]["b"].shape ==
           (1, agent_info["num_hidden_units"]))
    assert(test_agent.weights[1]["W"].shape ==
           (agent_info["num_hidden_units"], 1))
    assert(test_agent.weights[1]["b"].shape == (1, 1))

    print("weights[0][\"W\"]\n", (test_agent.weights[0]["W"]), "\n")
    print("weights[0][\"b\"]\n", (test_agent.weights[0]["b"]), "\n")
    print("weights[1][\"W\"]\n", (test_agent.weights[1]["W"]), "\n")
    print("weights[1][\"b\"]\n", (test_agent.weights[1]["b"]), "\n")

    agent_weight_answer = np.load("asserts/agent_init_weights_1.npz")
    assert(np.allclose(test_agent.weights[0]["W"], agent_weight_answer["W0"]))
    assert(np.allclose(test_agent.weights[0]["b"], agent_weight_answer["b0"]))
    assert(np.allclose(test_agent.weights[1]["W"], agent_weight_answer["W1"]))
    assert(np.allclose(test_agent.weights[1]["b"], agent_weight_answer["b1"]))

    print("Passed the asserts!")

    # Do not modify this cell!


def test_agent_start():
    ## Test Code for agent_start() ##

    agent_info = {"num_states": 500,
                  "num_hidden_layer": 1,
                  "num_hidden_units": 100,
                  "step_size": 0.1,
                  "discount_factor": 1.0,
                  "beta_m": 0.9,
                  "beta_v": 0.99,
                  "epsilon": 0.0001,
                  "seed": 10
                  }

    # Suppose state = 250
    state = 250

    test_agent = TDAgent()
    test_agent.agent_init(agent_info)
    test_agent.agent_start(state)

    print("Agent state: {}".format(test_agent.last_state))
    print("Agent selected action: {}".format(test_agent.last_action))

    assert(test_agent.last_state == 250)
    assert(test_agent.last_action == 1)

    print("Passed the asserts!")


def test_agent_step():
    # Do not modify this cell!

    ## Test Code for agent_step() ##
    agent_info = {"num_states": 5,
                  "num_hidden_layer": 1,
                  "num_hidden_units": 2,
                  "step_size": 0.1,
                  "discount_factor": 1.0,
                  "beta_m": 0.9,
                  "beta_v": 0.99,
                  "epsilon": 0.0001,
                  "seed": 0
                  }

    test_agent = TDAgent()
    test_agent.agent_init(agent_info)

    # load initial weights
    agent_initial_weight = np.load("asserts/agent_step_initial_weights.npz")
    test_agent.weights[0]["W"] = agent_initial_weight["W0"]
    test_agent.weights[0]["b"] = agent_initial_weight["b0"]
    test_agent.weights[1]["W"] = agent_initial_weight["W1"]
    test_agent.weights[1]["b"] = agent_initial_weight["b1"]

    # load m and v for the optimizer
    m_data = np.load("asserts/agent_step_initial_m.npz")
    test_agent.optimizer.m[0]["W"] = m_data["W0"]
    test_agent.optimizer.m[0]["b"] = m_data["b0"]
    test_agent.optimizer.m[1]["W"] = m_data["W1"]
    test_agent.optimizer.m[1]["b"] = m_data["b1"]

    v_data = np.load("asserts/agent_step_initial_v.npz")
    test_agent.optimizer.v[0]["W"] = v_data["W0"]
    test_agent.optimizer.v[0]["b"] = v_data["b0"]
    test_agent.optimizer.v[1]["W"] = v_data["W1"]
    test_agent.optimizer.v[1]["b"] = v_data["b1"]

    # Assume the agent started at State 3
    start_state = 3
    test_agent.agent_start(start_state)

    # Assume the reward was 10.0 and the next state observed was State 1
    reward = 10.0
    next_state = 1
    test_agent.agent_step(reward, next_state)

    # updated weights asserts
    print("updated_weights[0][\"W\"]\n", test_agent.weights[0]["W"], "\n")
    print("updated_weights[0][\"b\"]\n", test_agent.weights[0]["b"], "\n")
    print("updated_weights[1][\"W\"]\n", test_agent.weights[1]["W"], "\n")
    print("updated_weights[1][\"b\"]\n", test_agent.weights[1]["b"], "\n")

    agent_updated_weight_answer = np.load(
        "asserts/agent_step_updated_weights.npz")
    assert(np.allclose(test_agent.weights[0]
                       ["W"], agent_updated_weight_answer["W0"]))
    assert(np.allclose(test_agent.weights[0]
                       ["b"], agent_updated_weight_answer["b0"]))
    assert(np.allclose(test_agent.weights[1]
                       ["W"], agent_updated_weight_answer["W1"]))
    assert(np.allclose(test_agent.weights[1]
                       ["b"], agent_updated_weight_answer["b1"]))

    # last_state and last_action assert
    print("Agent last state:", test_agent.last_state)
    print("Agent last action:", test_agent.last_action, "\n")

    assert(test_agent.last_state == 1)
    assert(test_agent.last_action == 1)

    print("Passed the asserts!")


def test_agent_end():
    # Do not modify this cell!

    ## Test Code for agent_end() ##

    agent_info = {"num_states": 5,
                  "num_hidden_layer": 1,
                  "num_hidden_units": 2,
                  "step_size": 0.1,
                  "discount_factor": 1.0,
                  "beta_m": 0.9,
                  "beta_v": 0.99,
                  "epsilon": 0.0001,
                  "seed": 0
                  }

    test_agent = TDAgent()
    test_agent.agent_init(agent_info)

    # load initial weights
    agent_initial_weight = np.load("asserts/agent_end_initial_weights.npz")
    test_agent.weights[0]["W"] = agent_initial_weight["W0"]
    test_agent.weights[0]["b"] = agent_initial_weight["b0"]
    test_agent.weights[1]["W"] = agent_initial_weight["W1"]
    test_agent.weights[1]["b"] = agent_initial_weight["b1"]

    # load m and v for the optimizer
    m_data = np.load("asserts/agent_step_initial_m.npz")
    test_agent.optimizer.m[0]["W"] = m_data["W0"]
    test_agent.optimizer.m[0]["b"] = m_data["b0"]
    test_agent.optimizer.m[1]["W"] = m_data["W1"]
    test_agent.optimizer.m[1]["b"] = m_data["b1"]

    v_data = np.load("asserts/agent_step_initial_v.npz")
    test_agent.optimizer.v[0]["W"] = v_data["W0"]
    test_agent.optimizer.v[0]["b"] = v_data["b0"]
    test_agent.optimizer.v[1]["W"] = v_data["W1"]
    test_agent.optimizer.v[1]["b"] = v_data["b1"]

    # Assume the agent started at State 4
    start_state = 4
    test_agent.agent_start(start_state)

    # Assume the reward was 10.0 and reached the terminal state
    reward = 10.0
    test_agent.agent_end(reward)

    # updated weights asserts
    print("updated_weights[0][\"W\"]\n", test_agent.weights[0]["W"], "\n")
    print("updated_weights[0][\"b\"]\n", test_agent.weights[0]["b"], "\n")
    print("updated_weights[1][\"W\"]\n", test_agent.weights[1]["W"], "\n")
    print("updated_weights[1][\"b\"]\n", test_agent.weights[1]["b"], "\n")

    agent_updated_weight_answer = np.load(
        "asserts/agent_end_updated_weights.npz")
    assert(np.allclose(
        test_agent.weights[0]["W"], agent_updated_weight_answer["W0"]))
    assert(np.allclose(
        test_agent.weights[0]["b"], agent_updated_weight_answer["b0"]))
    assert(np.allclose(
        test_agent.weights[1]["W"], agent_updated_weight_answer["W1"]))
    assert(np.allclose(
        test_agent.weights[1]["b"], agent_updated_weight_answer["b1"]))

    print("Passed the asserts!")


test_get_value()

test_get_gradient()

test_agent_init()

test_agent_start()

test_agent_step()

test_agent_end()
