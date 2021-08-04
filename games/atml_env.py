import datetime
import os

import numpy
import torch

from .abstract_game import AbstractGame

#=====DeepLine imports===============
# TODO: check and change all deepLine imports - done
# TODO: check whether we need more classes to be copied from deepLine code to the atml_util folder - done
# TODO: check whether we need two renders (atml_util folder)
"""
deepline-v0 Environment:
    1. cursor moves from left to right
    2. inputs of each step (max of 2) must include the previous step of the same row
    3. Possible inputs for each step are (a) only previous step of the same row or (b) combination of previous step with
       one of the open list's steps
    4. open list can include only outputs of steps from higher rows and lower ranked families (families to the left of
       the step and above it.
    5. cell_options: a list of possible steps which are combinations of inputs from open list and primitives.
    6. options_windows: a chunk of cell_options currently on desplay for the agent to pick from
    7. Pipeline is calculated when agent reaches the final cell.
    8. Multiple step outputs are allowed
"""

import json
import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .atml_util.metrics import Accuracy

from .atml_util.primitives.data_preprocessing import *
from .atml_util.primitives.feature_preprocessing import *

from .atml_util.primitives.feature_eng_primitives import *

from .atml_util.primitives.feature_selection import *

from .atml_util.primitives.classifier_primitives import *
from .atml_util.primitives.regressor_primitives import *

from .atml_util.primitives.ensemble import *
# import primitives
from .atml_util.primitives import data_preprocessing, feature_eng_primitives, feature_selection, feature_preprocessing, \
        classifier_primitives, regressor_primitives, ensemble

from .atml_util.steps import Step
from .atml_util import steps
from .atml_util.pipelines import Pipeline, Pipeline_run
from .atml_util import LearningJob
from .atml_util import train_test_evaluation
from .atml_util import ML_Render, rgb_render
from .atml_util.equal_groups import EqualGroupsKMeans

import numpy as np
import pandas as pd
from itertools import cycle
from random import Random
import random
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
import math
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)

LjRandom = Random(356)
shRandom = Random(111)
npRandom = RandomState(234)


primtive_modules = {
    'data preprocess': data_preprocessing,
    'feature preprocess': feature_preprocessing,
    'feature selection': feature_selection,
    'feature engineering': feature_eng_primitives,
    'Prediction': classifier_primitives,
    'ensemble': ensemble
}

primitives = {
    'data preprocess': [Imputer, OneHotEncoderPrim, LabelEncoderPrim, ImputerIndicatorPrim, NumericDataPrim,
                        ImputerEncoderPrim, ImputerMedian, ImputerOneHotEncoderPrim],
    'feature preprocess': [KBinsDiscretizerOneHotPrim, NormalizerPrim, PowerTransformerPrim,
                           QuantileTransformerPrim, RobustScalerPrim, MaxAbsScalerPrim, MinMaxScalerPrim,
                           KBinsDiscretizerOrdinalPrim, StandardScalerPrim],
    'feature selection': [VarianceThresholdPrim, UnivariateSelectChiKbestPrim, f_classifKbestPrim,
    mutual_info_classifKbestPrim, f_regressionKbestPrim, f_classifPercentilePrim, f_regressionPercentilePrim,
    mutual_info_classifPercentilePrim, mutual_info_regressionKbestPrim, mutual_info_regressionPercentilePrim,
    UnivariateSelectChiPercentilePrim, RFE_RandomForestPrim, RFE_GradientBoostingPrim, RFE_RandomForestRegPrim,
    RFE_SVRPrim, UnivariateSelectChiFDRPrim, UnivariateSelectChiFWEPrim, UnivariateSelectChiFPRPrim, f_classifFDRPrim,
    f_regressionFPRPrim, f_classifFPRPrim, f_classifFWEPrim,
    f_regressionFDRPrim, f_regressionFWEPrim],
    'feature engineering': [PCA_LAPACK_Prim, PCA_ARPACK_Prim, InteractionFeaturesPrim, PolynomialFeaturesPrim,
                            IncrementalPCA_Prim, PCA_Randomized_Prim, KernelPCA_Prim,
                            TruncatedSVD_Prim, FastICA_Prim, RandomTreesEmbeddingPrim],
    'Prediction': [RandomForestClassifierPrim, AdaBoostClassifierPrim, BaggingClassifierPrim,
                       BernoulliNBClassifierPrim, ComplementNBClassifierPrim, DecisionTreeClassifierPrim,
                       ExtraTreesClassifierPrim, GaussianNBClassifierPrim, GaussianProcessClassifierPrim,
                       GradientBoostingClassifierPrim, KNeighborsClassifierPrim, LinearDiscriminantAnalysisPrim,
                       LinearSVCPrim, LogisticRegressionPrim, LogisticRegressionCVPrim, MultinomialNBPrim,
                       NearestCentroidPrim, PassiveAggressiveClassifierPrim,
                       QuadraticDiscriminantAnalysisPrim, RidgeClassifierPrim,
                       RidgeClassifierCVPrim, SGDClassifierPrim, SVCPrim, XGBClassifierPrim,
                       BalancedRandomForestClassifierPrim, EasyEnsembleClassifierPrim, RUSBoostClassifierPrim,
                       LGBMClassifierPrim, ARDRegressionPrim, AdaBoostRegressorPrim, BaggingRegressorPrim],
    'ensemble': [MajorityVotingPrim, RandomForestMetaPrim, RandomForestRegressorMetaPrim, AdaBoostClassifierMetaPrim,
                 ExtraTreesClassifierMetaPrim, GradientBoostingClassifierMetaPrim, XGBClassifierMetaPrim]
}

families = {
    'data preprocess': 1,
    'feature preprocess': 2,
    'feature selection': 3,
    'feature engineering': 4,
    'Classifier': 5,
    'Regressor': 6,
    'ensemble': 7
}

all_primitives = []
for val in primitives.values():
    all_primitives += val

num_primitives = len(all_primitives) + 2


# TODO: check where we can move all the code from line 98 to 255 - some global init() function like MuZeroConfig?

class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (1, 1,
                                  9)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(2))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 6  # Maximum number of moves if game is not finished before
        self.num_simulations = 10  # Number of future moves self-simulated
        self.discount = 0.978  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 5
        self.fc_representation_layers = [16]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network

        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results",
                                         os.path.basename(__file__)[:-3], datetime.datetime.now().strftime(
                "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 30000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 32  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.0064  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.replay_buffer_size = 5000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 7  # Number of game moves to keep for every batch element
        self.td_steps = 7  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0.2  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = AutomlEnv(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return [[observation]], reward * 10, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        self.env.get_cell_options()
        return self.env.cell_options
    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return [[self.env.reset()]]

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Down",
            1: "Right",
        }
        return f"{action_number}. {actions[action_number]}"


class GridEnv:
    def __init__(self, size=3):
        self.size = size
        self.position = [0, 0]

    def legal_actions(self):
        legal_actions = list(range(2))
        if self.position[0] == (self.size - 1):
            legal_actions.remove(0)
        if self.position[1] == (self.size - 1):
            legal_actions.remove(1)
        return legal_actions

    def step(self, action):
        if action not in self.legal_actions():
            pass
        elif action == 0:
            self.position[0] += 1
        elif action == 1:
            self.position[1] += 1

        reward = 1 if self.position == [self.size - 1] * 2 else 0
        return self.get_observation(), reward, bool(reward)

    def reset(self):
        self.position = [0, 0]
        return self.get_observation()

    def render(self):
        im = numpy.full((self.size, self.size), "-")
        im[self.size - 1, self.size - 1] = "1"
        im[self.position[0], self.position[1]] = "x"
        print(im)

    def get_observation(self):
        observation = numpy.zeros((self.size, self.size))
        observation[self.position[0]][self.position[1]] = 1
        return observation.flatten()

#==========From deepLine====================================


'''DeepLine observation class'''
class Observation:
    def __init__(self, level=1, mode='train', seed=0):
        # property related to the grid depth level init = 1 - verify?
        self.level = level
        #
        self.grid_families = [['data preprocess', 'feature preprocess', 'feature selection', 'feature engineering', 'Prediction', 'ensemble']]*level
        # list of all primitives
        self.all_primitives = all_primitives

        # dictionary of primitives divided to categories
        self.primitives = primitives
        #
        self.num_primitives = len(self.all_primitives) + 2
        # grid initiation
        self.grid = np.full((level, 6), 'BLANK').tolist()   # populate with steps
        # index of the last primitive in each raw
        self.last_in_rows = np.full((level), -1).tolist()
        self.pipe_run = None
        # cursor place
        self.cursor = [0, 0]
        # current learning job - verify?
        self.learning_job = None
        # load all learning jobs by criteria
        self.all_learning_jobs = LearningJob.load_all_learning_jobs(mode=mode, metric=Accuracy())
        # loaded list with learning job objects
        self.curr_learning_jobs = list(self.all_learning_jobs.values())
        # next learning job - if we want to proceed with some fixed order
        self.next_lj = cycle(self.curr_learning_jobs)
        # all available outputs on the grid - verify?
        self.open = []  # Dict of all pipeline's steps outputs
        # all available input for current cell - verify?
        self.cell_options = []  # list of all possible step inputs for current cell
        # possible length for the output to input connection
        self.window_size = len(self.grid[0])
        # max possible inputs
        self.max_inputs = 2
        # existing edges in the grid
        self.relations = []
        # existing windows - verify?
        self.options_windows = []  # Windows in size window_size showing only steps from cell_options in window
        self.window_index = 0
        self.X_train = self.X_test = self.Y_train = self.Y_test = None
        self.last_reward = None
        # next level in the hierarchy
        self.next_level = []
        # current hierarchical step level
        self.hier_level = 1
        # for the agent - if agent need to save this observation
        self.register_state = False
        # each step to relevant input
        self.input_to_cell_dict = {0: np.zeros(2)-1}
        # last cells that we dont need for combiner
        self.skip_cells = [[i, len(self.grid[0])-1] for i in range(self.level - 1)]
        # metafeatures as last part of the pipeline
        self.num_estimators = 0
        self.best_pipeline = None
        self.best_score = 0
        self.cv_reward = False
        self.print_scores = False
        self.prev_output = []
        self.split_rate = 0.8
        self.random_state = seed
        # not used - reward if it is better than baseline the reward will be higher
        self.baselines_rewards = False
        self.model = None
        self.log_pipelines = False
        self.reset_observation()

    def chunks(self, l, n):

        if self.model:
            steps_matrix = []
            for stp in l:
                if stp == 'BLANK':
                    stp_inputs = np.zeros(self.level) - 1
                    stp_prim = np.array([self.num_primitives - 2])
                    stp_mf = np.zeros(num_metafeatures)

                elif stp == 'FINISH':
                    stp_inputs = np.zeros(self.level) - 1
                    stp_prim = np.array([self.num_primitives - 1])
                    stp_mf = np.zeros(num_metafeatures)
                else:
                    stp_inputs = stp.vec_representation[0]
                    stp_prim = stp.vec_representation[2]
                    stp_mf = stp.vec_representation[1]
                steps_matrix.append(self.model.get_actions_vec(stp_prim, stp_inputs, stp_mf).tolist())

            n_clusters = math.ceil(len(l)/n)

            # if len(l) <= n_clusters
            steps_matrix = np.array(steps_matrix)
            clf = EqualGroupsKMeans(n_clusters=n_clusters, random_state=self.random_state)
            clf.fit(steps_matrix)

            for lbl in np.unique(clf.labels_):
                inds = np.where(clf.labels_ == lbl)
                cluster = [l[i] for i in inds[0]]
                if len(inds[0]) < n:
                    cluster += (n - len(inds[0])) * [-1]
                yield cluster

        else:
            # For item i in a range that is a length of l,
            if not len(l) % n == 0:
                complete_len = len(l) + (n - len(l) % n)
                l += (complete_len - len(l)) * [-1]
            # shRandom.seed(0)
            shRandom.shuffle(l)
            for i in range(0, len(l), n):
                # Create an index range for l of n items:
                yield l[i:i + n]

    def get_all_learning_jobs(self):
        return [val.name for val in self.all_learning_jobs.values()]

    def reset_observation(self, primitives_list=None, lj_indices=None):
        # shRandom = Random(111)
        # npRandom = RandomState(234)
        if primitives_list:
            self.all_primitives = [prmtv for prmtv in self.all_primitives if prmtv().name in primitives_list]
            self.num_primitives = len(self.all_primitives) + 2
            for family, fami_prims in self.primitives.items():
                self.primitives[family] = [prmtv for prmtv in fami_prims if prmtv().name in primitives_list]
        if lj_indices:
            self.curr_learning_jobs = [list(self.all_learning_jobs.values())[index] for index in lj_indices]
            # self.next_lj = np.random.choice(self.curr_learning_jobs)

        self.last_in_rows = np.full((self.level), -1).tolist()
        self.learning_job = LjRandom.choice(self.curr_learning_jobs)
        self.cell_options = []
        self.options_windows = []
        self.relations = []
        self.open = []  # Dict of all pipeline's steps outputs
        self.pipe_run = Pipeline_run(self.learning_job)
        self.cursor = [0, 0]
        self.input_to_cell_dict = {0: np.zeros(2)-1}
        self.grid = np.full((self.level, 6), 'BLANK').tolist()
        # task = self.learning_job.task
        # for i in range(self.level):
        # self.grid_families[0][4] = task

        if self.split_rate == 1:
            self.X_train = self.learning_job.dataset.X.copy(deep=True)
            self.X_test = self.Y_test = pd.DataFrame()
            self.Y_train = np.copy(self.learning_job.dataset.Y)
        else:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.learning_job.dataset.X.copy(deep=True),
                                                                                   np.copy(self.learning_job.dataset.Y),
                                                                                   train_size=self.split_rate, test_size=1-self.split_rate,
                                                                                   random_state=self.random_state)
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.Y_train = self.Y_train
        self.pipe_run.fit(self.X_train, self.Y_train)
        self.get_cell_options()
        self.options_windows = list(self.chunks(self.cell_options, self.window_size))
        self.window_index = 0
        self.next_level = []
        self.hier_level = 1
        self.relations = []
        self.last_reward = 0
        self.last_output_vec = generate_metafeatures(self.pipe_run.fit_outputs[0][0], use_correlation=True)
        self.num_estimators = 0

    def gentr_fn(self, alist):
        while 1:
            for j in alist:
                yield j

    def get_open(self):
        self.open = []
        if self.cursor[1] == len(self.grid[0]) - 1 and not self.cursor[0] == len(self.grid) - 1:  # ensemble cell
            self.open = []
            return

        if self.cursor[1] == len(self.grid[0]) - 1 and self.cursor[0] == len(self.grid) - 1:  # ensemble cell
            for step in self.last_in_rows:
                if step == -1:
                    continue
                if not step.primitive.type == 'Classifier' and not step.primitive.type == 'Regressor':
                    continue
                else:
                    self.open.append([step.index, 0])
            self.num_estimators = len(self.open)
            redundant = self.get_redundant()
            if len(redundant) > 0:
                self.open += redundant
            return

        for i in range(self.cursor[0]):
            for j in range(self.cursor[1] + 1):
                if self.cursor[1] == len(self.grid[0]) - 1 and not j == self.cursor[1] - 1:
                    continue
                if self.grid[i][j] == 'BLANK' or self.grid[i][j] == 'FINISH':
                    continue
                step_index = self.grid[i][j].index
                outputs = self.pipe_run.fit_outputs[step_index]
                for output, _ in outputs.items():
                    self.open.append([step_index, output])# , output

    def get_cell_options(self):

        if self.cursor[1] == len(self.grid[0]) - 1 and not self.cursor[0] == len(self.grid) - 1:  # ensemble cell
            self.cell_options = ['BLANK', 'FINISH']
            return
        last_in_row = [[0, 0]]
        # data = dict()
        # print('Starts creating steps open list')
        for i in range(self.cursor[1]):  # find last step in the cursor's row and add its outputs - change with last_in_rows
            if not self.grid[self.cursor[0]][i] == 'BLANK' or self.grid[self.cursor[0]][i] == 'FINISH':
                step_index = self.grid[self.cursor[0]][i].index
                outputs = self.pipe_run.fit_outputs[step_index]
                s = []
                for output, _ in outputs.items():
                    s.append([step_index, output])
                last_in_row = s

        family_primitives = self.primitives[self.grid_families[self.cursor[0]][self.cursor[1]]]
        family_module = primtive_modules[self.grid_families[self.cursor[0]][self.cursor[1]]]
        all_possible_inputs = list(itertools.product(last_in_row, self.open))  # All combinations of outputs of open and outputs of last step in row

        all_possible_inputs = list(map(list, all_possible_inputs)) + [list(map(list, last_in_row))]  # + open_list
        if self.cursor[1] == len(self.grid[0]) - 1 and self.cursor[0] == len(self.grid) - 1:
            all_possible_inputs = [self.open]
        # This only allows for a max of 2 inputs! maybe change in future

        for inputs in all_possible_inputs:
            i = 0
            data = dict()
            for input_idx in inputs:
                data[i] = self.pipe_run.fit_outputs[input_idx[0]][input_idx[1]]
                i += 1
            data = family_module.handle_data(data)
            if not data:
                self.cell_options = []
            else:
                if data['X'].empty or data['X'].shape[1] == 0:
                    continue

                acceptance_dict = {}
                data_vec = generate_metafeatures(data)
                for primitive in family_primitives:
                    if not primitive in self.all_primitives:
                        continue
                    if primitive().accept_type in acceptance_dict:
                        accepts = acceptance_dict[primitive().accept_type]
                    else:
                        accepts = primitive().can_accept(data)
                        acceptance_dict[primitive().accept_type] = accepts
                    if accepts and primitive().is_needed(data):
                        step = Step(len(self.pipe_run.steps) + 1, inputs, primitive(random_state=self.random_state), data_vec)
                        ind = self.all_primitives.index(primitive)
                        step.to_vector(self.num_primitives, ind, self.level)
                        self.cell_options.append(step)
        if not self.cursor == [len(self.grid) - 1, len(self.grid[0]) - 1]:
            self.cell_options.append('BLANK')
        elif len(self.cell_options) == 0:
            self.cell_options.append('BLANK')
        elif self.num_estimators == 1:
            self.cell_options = ['BLANK']
        # print('Stops creating steps open list')

    def add_step(self, step):
        last_step = len(self.pipe_run.steps)
        assert last_step + 1 == step.index
        self.grid[self.cursor[0]][self.cursor[1]] = step
        self.last_in_rows[self.cursor[0]] = step
        step_output = self.pipe_run.add_step(step)
        self.input_to_cell_dict[step.index] = np.array(self.cursor)

    def get_state(self, actions_rep=True):
        lj_vector = self.learning_job.to_vector()
        steps_inputs = np.ndarray(0)
        steps_prim = np.ndarray(0)
        options_family = np.zeros(self.window_size)
        s = self.options_windows[self.window_index]

        i = 0
        for stp in self.options_windows[self.window_index]:
            if stp == 'BLANK':
                stp_inputs = np.zeros(self.level) - 1
                stp_prim = np.array([self.num_primitives-2])
                options_family[i] = 8

            elif stp == 'FINISH':
                stp_inputs = np.zeros(self.level) - 1
                stp_prim = np.array([self.num_primitives - 1])
                options_family[i] = 9

            elif stp == -1:
                stp_inputs = np.zeros(self.level) - 1
                stp_prim = np.array([-1])  # remember
                options_family[i] = 0
            else:
                stp_inputs = stp.vec_representation[0]
                stp_prim = stp.vec_representation[2]
                options_family[i] = families[stp.primitive.type]
            i += 1
            steps_inputs = np.concatenate((steps_inputs, stp_inputs))
            steps_prim = np.concatenate((steps_prim, stp_prim))

        self.relations = np.empty(0)
        grid_primitives_vec = np.empty(0)

        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                # if [i, j] in self.skip_cells:
                #     continue
                inputs_vec = np.zeros(self.level * 2) - 99
                stp = self.grid[i][j]
                if i == self.cursor[0] and j == self.cursor[1]:  # if real step under cursor - ignore (lower hierarchy)
                    stp = 'BLANK'
                # family = self.grid_families[i][j]
                # cell_vec = np.zeros(len(self.primitives[family]) + 1)
                cell_vec = np.zeros(1)
                if stp == 'BLANK' or stp == 'FINISH':
                    self.relations = np.concatenate((self.relations, inputs_vec))
                    if stp == 'BLANK':
                        cell_vec[0] = self.num_primitives - 2
                    if stp == 'FINISH':
                        cell_vec[0] = self.num_primitives - 1
                    grid_primitives_vec = np.concatenate((grid_primitives_vec, cell_vec))
                else:
                    inputs = [v[0] for v in stp.input_indices]
                    k = 0
                    for z in inputs:
                        inputs_vec[k], inputs_vec[k+1] = self.input_to_cell_dict[z][0], self.input_to_cell_dict[z][1]
                        k += 2
                    self.relations = np.concatenate((self.relations, inputs_vec))
                    # ind = self.all_primitives.index(stp.primitive.__class__)
                    cell_vec = stp.vec_representation[2]
                    grid_primitives_vec = np.concatenate((grid_primitives_vec, cell_vec))


        if actions_rep:
            state_vec = np.concatenate((grid_primitives_vec, self.relations, np.array(self.cursor), pipeline_metadata, lj_vector, options_family, steps_prim, steps_inputs))
        else:
            state_vec = np.concatenate((grid_primitives_vec, lj_vector, np.array(self.cursor), self.relations, pipeline_metadata))


        info = {}
        info['cells_num'] = len(self.grid[0]) * self.level  # - len(self.skip_cells)
        info['grid_prims_size'] = len(grid_primitives_vec)  # size of primitives on grid representation
        info['num_prims'] = self.num_primitives
        info['relations_size'] = len(self.relations)  # size of relations vector
        info['single_relation_size'] = int(self.level * 2)
        info['ff_state_size'] = len(np.array(self.cursor)) + len(pipeline_metadata) +\
                                len(lj_vector) + len(options_family)  # rest of state vector
        info['action_prims'] = len(steps_prim)
        info['action_inputs'] = len(steps_inputs)
        info['max_inputs'] = self.level

        return state_vec, info

    def get_reward(self, done):
        if not done:
            reward = 0
        else:
            self.pipe_run.refit()
            if 'predictions' in self.pipe_run.fit_outputs.keys():
                if not self.cv_reward:
                    self.pipe_run.produce(self.X_test)
                    score = self.learning_job.metric.evaluate(self.Y_test, self.pipe_run.produce_outputs['predictions'])
                    vec = self.meta_regressor_state.tolist()
                    vec.append(score)
                    self.meta_regressor_data.append(vec)
                else:
                    score = self.learning_job.metric.cv_evaluate(self.X_train, self.Y_train, deepcopy(self.pipe_run))
                if self.log_pipelines:
                    self.pipe_run.log_to_json(score)
                if not self.baselines_rewards:
                    reward = score
                else:
                    reward = (self.learning_job.base_scores <= score).mean()
                if score >= self.best_score:
                    self.best_score = score
                    self.best_pipeline = self.pipe_run
                if self.print_scores:
                    print(self.learning_job.name + ' - achieved: ' + str(score))
            else:
                reward = -1
        return reward

    def move_cursor(self, finish=False):
        # if self.cursor == [2, 4]:
        #     print('debug')
        if self.cursor == [len(self.grid)-1, len(self.grid[0])-1]:
            return True
        elif finish:
            self.cursor = [len(self.grid) - 1, len(self.grid[0]) - 1]
        elif not self.cursor[1] == len(self.grid[0])-1:
            self.cursor[1] += 1
        else:
            self.cursor[1] = 0
            self.cursor[0] += 1
        # if self.cursor in self.skip_cells and not finish:
        #     self.move_cursor()
        self.cell_options = []
        self.open = []
        self.get_open()
        self.get_cell_options()
        self.options_windows = list(self.chunks(self.cell_options, self.window_size))
        self.window_index = 0
        self.next_level = []
        self.hier_level = 1
        return False

    # connect all sub pipelines to the combiner
    def get_redundant(self):
        redundant = []
        for row in self.grid:
            for step in row:
                if step == 'BLANK' or step == 'FINISH':
                    continue
                if step.primitive.type == 'Classifier' or step.primitive.type == 'Regressor':
                    continue
                if self.pipe_run.is_redundant(step.index):
                    redundant.append([step.index, 0])
        return redundant

    def get_base_pipelines(self):
        baselines = [RandomForestClassifierPrim, ExtraTreesClassifierPrim, XGBClassifierPrim]
        pipelines = {}
        for baseline in baselines:
            base_pipeline = Pipeline_run(learning_job=self.learning_job)
            first = Step(1, [[0, 0]], ImputerOneHotEncoderPrim())
            sec = Step(2, [[1, 0]], baseline())
            base_pipeline.steps.append(first)
            base_pipeline.steps.append(sec)
            pipelines[baseline().name] = base_pipeline
        return pipelines

    def compute_baselines(self):
        baselines = [RandomForestClassifierPrim, ExtraTreesClassifierPrim, XGBClassifierPrim, LinearSVCPrim,
                     LogisticRegressionPrim, GradientBoostingClassifierPrim, KNeighborsClassifierPrim,
                     DecisionTreeClassifierPrim]

        path = 'base_score_{}_{}.json'.format(self.cv_reward, self.split_rate)
        if os.path.isfile(path):
            with open(path) as json_data:
                all_scores = json.load(json_data)
        else:
            all_scores = {}
        for lj in self.curr_learning_jobs:
            if lj.name in all_scores:
                lj.base_scores = np.array(all_scores[lj.name])
                continue
            scores = []
            for base in baselines:
                base_pipeline = Pipeline_run(lj)
                first = Step(1, [[0, 0]], ImputerOneHotEncoderPrim())
                sec = Step(2, [[1, 0]], base(random_state=self.random_state))
                base_pipeline.steps.append(first)
                base_pipeline.steps.append(sec)

                if self.split_rate == 1:
                    x_train = lj.dataset.X.copy(deep=True).reset_index(drop=True)
                    x_test = y_test = pd.DataFrame()
                    y_train = np.copy(lj.dataset.Y)
                else:
                    x_train, x_test, y_train, y_test = train_test_split(
                        lj.dataset.X.copy(deep=True),
                        np.copy(lj.dataset.Y),
                        train_size=self.split_rate, test_size=1 - self.split_rate,
                        random_state=self.random_state)
                    x_train = x_train.reset_index(drop=True)
                    x_test = x_test.reset_index(drop=True)

                if not self.cv_reward:
                    base_pipeline.fit(x_train, y_train)
                    base_pipeline.produce(x_test)
                    score = lj.metric.evaluate(y_test, base_pipeline.produce_outputs['predictions'])
                else:
                    score = lj.metric.cv_evaluate(x_train, y_train, deepcopy(base_pipeline))
                if score > 0:
                    scores.append(score)
            all_scores[lj.name] = scores
            print(lj.name + ': ' + str(scores))
            lj.base_scores = np.array(scores)
        with open(path, 'w') as f:
            json.dump(all_scores, f, indent=2)


'''DeepLine environment'''
class AutomlEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed):
        self.mode = 'train'
        self.observation = Observation(level=3, mode=self.mode, seed=self.seed)
        self.observation.reset_observation()
        arr = self.observation.get_state()[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=arr.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(self.observation.window_size)
        self.first_render = True
        self.rendition = None
        self.last_action = None
        self.main_loop = None
        self.heirarc_step = True
        self.actions_dict = None
        self.steps_dict = {}
        self.state_info = None
        # self.model = None
        self.embedd_size = None
        self.seed = seed

    # TODO: same as get_observation() - check if we want to leave the same name
    def get_state(self):
        if not self.heirarc_step:
            state, _ = self.observation.get_state(actions_rep=False)
            act_vec = []
            for k in self.actions_dict.values():
                if self.steps_dict[k]:
                    act_vec += [1]
                else:
                    act_vec += [0]
            act_vec = np.array(act_vec)
            state = np.concatenate((state, act_vec))
        else:
            state, self.state_info = self.observation.get_state()

            if self.embedd_size:
                self.state_info['step_size'] = self.embedd_size + num_metafeatures + self.observation.level
                self.state_info['processed_actions_size'] = self.action_space.n * (self.state_info['step_size'])
            if self.observation.model:
                state = self.observation.model.process_state_vec(state, self.state_info)
        return state

    # ===Internal
    # not needed only for comparison with hierarchical step
    def regular_step(self, action):
        action_key = self.actions_dict[action]
        step_action = self.steps_dict[action_key]

        if step_action:
            if step_action == 'BLANK':
                self.observation.grid[self.observation.cursor[0]][self.observation.cursor[1]] = 'BLANK'
            elif step_action == 'FINISH':
                self.observation.grid[self.observation.cursor[0]][self.observation.cursor[1]] = 'FINISH'
                done = self.observation.move_cursor(True)
                state = self.get_state()
                self.observation.last_reward = self.observation.get_reward(done)
                self.observation.register_state = True
                if self.observation.last_reward > 0:
                    print(self.observation.last_reward)
                self.steps_dict = self.steps_dict.fromkeys(self.steps_dict, None)
                for step in self.observation.cell_options:
                    if step == -1:
                        continue
                    if step == 'BLANK' or step == 'FINISH':
                        self.steps_dict[step] = step
                    else:
                        ipt = str([self.observation.input_to_cell_dict[item[0]].astype(int).tolist() for item in
                                   step.input_indices])
                        step_key = str([ipt, step.primitive.name])
                        if len(self.steps_dict) > 0 and not step_key in self.steps_dict:
                            raise Exception('step not in dict')
                        self.steps_dict[step_key] = step
                return state, self.observation.last_reward, done, {'episode': None,
                                                                   'register': self.observation.register_state}
            else:
                self.observation.add_step(step_action)
            done = self.observation.move_cursor()
            state = self.get_state()
            self.observation.last_reward = self.observation.get_reward(done)
            if self.observation.last_reward > 0:
                print(self.observation.last_reward)
            self.steps_dict = self.steps_dict.fromkeys(self.steps_dict, None)
            for step in self.observation.cell_options:
                if step == -1:
                    continue
                if step == 'BLANK' or step == 'FINISH':
                    self.steps_dict[step] = step
                else:
                    ipt = str([self.observation.input_to_cell_dict[item[0]].astype(int).tolist() for item in
                               step.input_indices])
                    step_key = str([ipt, step.primitive.name])
                    if len(self.steps_dict) > 0 and not step_key in self.steps_dict:
                        raise Exception('step not in dict')
                    self.steps_dict[step_key] = step
            return state, self.observation.last_reward, done, {'episode': None, 'register': self.observation.register_state,
                                                                   'hier_level': self.observation.hier_level}
        else:
            self.observation.grid[self.observation.cursor[0]][self.observation.cursor[1]] = 'BLANK'
            done = False
            state = self.get_state()
            self.observation.last_reward = -1
            self.steps_dict = self.steps_dict.fromkeys(self.steps_dict, None)
            for step in self.observation.cell_options:
                if step == -1:
                    continue
                if step == 'BLANK' or step == 'FINISH':
                    self.steps_dict[step] = step
                else:
                    ipt = str([self.observation.input_to_cell_dict[item[0]].astype(int).tolist() for item in
                               step.input_indices])
                    step_key = str([ipt, step.primitive.name])
                    if len(self.steps_dict) > 0 and not step_key in self.steps_dict:
                        raise Exception('step not in dict')
                    self.steps_dict[step_key] = step
            return state, self.observation.last_reward, done, {'episode': None, 'register': self.observation.register_state,
                                                                   'hier_level': self.observation.hier_level}

    #===Internal
    def hierarchical_step(self, action):
        done = False
        if len(self.observation.options_windows) == 0 or len(self.observation.options_windows[0]) == 0:
            state = self.get_state()
            self.observation.last_reward = -1
            self.observation.register_state = False
            done = True
            return state, self.observation.last_reward, done, {'episode': None, 'register': self.observation.register_state,
                                                                   'hier_level': self.observation.hier_level}

        step = self.observation.options_windows[self.observation.window_index][action]

        if step == -1:  # Invalid action
            self.observation.last_reward = -1
            done = False
            self.observation.register_state = False
            # shRandom.seed(0)
            shRandom.shuffle(self.observation.options_windows[self.observation.window_index])
            state = self.get_state()
            return state, self.observation.last_reward, done, {'episode': None, 'register': self.observation.register_state,
                                                                   'hier_level': self.observation.hier_level}

        # Else, regular case:
        if len(self.observation.options_windows) == 1:
            step = self.observation.options_windows[self.observation.window_index][action]
            if step == 'BLANK':
                self.observation.grid[self.observation.cursor[0]][self.observation.cursor[1]] = 'BLANK'
            elif step == 'FINISH':
                self.observation.grid[self.observation.cursor[0]][self.observation.cursor[1]] = 'FINISH'
                hlevel = self.observation.hier_level
                done = self.observation.move_cursor(True)
                state = self.get_state()
                self.observation.last_reward = self.observation.get_reward(done)
                self.observation.register_state = True
                return state, self.observation.last_reward, done, {'episode': None, 'register': self.observation.register_state,
                                                                   'hier_level': hlevel}
            else:
                self.observation.add_step(step)
            hlevel = self.observation.hier_level
            done = self.observation.move_cursor()
            state = self.get_state()
            self.observation.last_reward = self.observation.get_reward(done)
            self.observation.register_state = True
            return state, self.observation.last_reward, done, {'episode': None, 'register': self.observation.register_state,
                                                                   'hier_level': hlevel}

        self.observation.grid[self.observation.cursor[0]][self.observation.cursor[1]] = step
        self.observation.next_level.append(step)
        self.observation.window_index += 1
        hlevel = self.observation.hier_level
        # state = self.get_state()
        if self.observation.window_index == len(self.observation.options_windows):
            temp = list(self.observation.chunks(self.observation.next_level, self.observation.window_size))
            if len(temp) == 1 and len(self.observation.options_windows) < 5:
                l1 = [num for elem in self.observation.options_windows for num in elem]
                flattened_options = [item for item in l1 if item not in temp[0]]
                indexes = [i for i, x in enumerate(temp[0]) if x == -1]
                n = 5 - len(self.observation.options_windows)
                # np.random.seed(0)
                # npRandom.seed(0)
                picked = npRandom.choice(flattened_options, n, replace=False)
                indices = npRandom.choice(indexes, n, replace=False)
                j = 0
                for i in indices:
                    temp[0][i] = picked[j]
                    j += 1
                    if not len(list(set(temp[0]) - set(temp[0]))) == 0:
                        print('PROBLEM')
            self.observation.options_windows = temp
            self.observation.next_level = []
            self.observation.hier_level += 1
            self.observation.window_index = 0
        state = self.get_state()
        self.observation.register_state = False
        return state, self.observation.last_reward, done, {'episode': None, 'register': self.observation.register_state,
                                                           'hier_level': hlevel}

    #===OK
    def step(self, action):
        if action > self.action_space.n:
            raise ValueError('Invalid action!')
        self.last_action = action

        try:
            if not self.first_render:
                self.rendition.reset(self.observation, action=self.last_action)

            if self.heirarc_step:
                return self.hierarchical_step(action)
            else:
                return self.regular_step(action)
        except Exception as e:  # Check if handled correctly
            if not 'sklearn error in FactICA (array must not contain infs or NaNs)- skipping the primitive' in e.args:
                print(e)
            if isinstance(self.observation.grid[self.observation.cursor[0]][self.observation.cursor[1]], Step):
                self.observation.pipe_run.rm_last_step(self.observation.grid[self.observation.cursor[0]][self.observation.cursor[1]].index)
                row = self.observation.cursor[0]
                self.observation.last_in_rows[row] = -1
                for stp in reversed(self.observation.grid[row]):
                    if isinstance(stp, Step) and stp != self.observation.grid[self.observation.cursor[0]][self.observation.cursor[1]]:
                        self.observation.last_in_rows[row] = stp
            self.observation.grid[self.observation.cursor[0]][self.observation.cursor[1]] = 'BLANK'
            print(self.observation.cursor)
            hlevel = self.observation.hier_level
            done = self.observation.move_cursor()
            state = self.get_state()
            self.observation.last_reward = -1
            self.observation.register_state = True
            return state, self.observation.last_reward, done, {'episode': None, 'register': self.observation.register_state,
                                                               'hier_level': hlevel}

    # TODO: to check why not in init()
    def set_env_params(self, primitives_list=None, lj_list=None, cv_reward=False, print_scores=False, level=3,
                       reset_regressor=True, split_rate=0.8, baselines_rewards=False,
                       heirarc_step=True, embedd_size=None, log_pipelines=False):

        self.heirarc_step = heirarc_step
        self.observation = Observation(level=level, mode=self.mode)
        self.observation.split_rate = split_rate
        self.observation.random_state = self.seed
        self.observation.split_rate = split_rate
        self.observation.reset_observation(primitives_list, lj_list)
        self.observation.cv_reward = cv_reward
        self.observation.baselines_rewards = baselines_rewards
        self.observation.log_pipelines = log_pipelines
        if baselines_rewards:
            self.observation.compute_baselines()
        self.observation.print_scores = print_scores
        if reset_regressor:
            self.observation.meta_regressor_data = []
        if not heirarc_step:
            self.actions_dict, self.steps_dict = self.get_actions_dict()
            self.action_space = spaces.Discrete(len(self.actions_dict))
            for step in self.observation.cell_options:
                if step == -1:
                    continue
                if step == 'BLANK' or step == 'FINISH':
                    self.steps_dict[step] = step
                else:
                    ipt = str([self.observation.input_to_cell_dict[item[0]].astype(int).tolist() for item in
                               step.input_indices])
                    step_key = str([ipt, step.primitive.name])
                    if len(self.steps_dict) > 0 and not step_key in self.steps_dict:
                        raise Exception('step not in dict')
                    self.steps_dict[step_key] = step
        arr = self.get_state()
        shape = arr.shape
        if embedd_size:
            self.embedd_size = embedd_size
            shape = (shape[0] - self.state_info['action_prims'] + self.state_info['action_prims'] * embedd_size,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)  # Change!
        self.reset()

    #===OK
    def reset(self):
        self.observation.reset_observation()
        return self.get_state()

    #===OK
    def render(self, mode='human', close=False):
        if mode == 'human':
            if self.first_render:
                self.rendition = ML_Render.MLGrid(self.observation)
                self.rendition.generate_grid()
                self.first_render = False
            else:
                self.rendition.reset(self.observation, self.last_action)
                self.main_loop = self.rendition.canvas.update()
        else:
            if self.first_render:
                self.rendition = rgb_render.MLGrid(self.observation)
                rgb_arr = self.rendition.generate_grid()
                self.first_render = False
                return rgb_arr
            else:
                rgb_arr = self.rendition.reset(self.observation, self.last_action)
                return rgb_arr


