# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: wangronin
"""

from __future__ import division
from __future__ import print_function

import pdb
import warnings, dill, functools, itertools
from joblib import Parallel, delayed
import copyreg as copy_reg
#import copy_reg
import queue
import threading

import pandas as pd
import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import r2_score
import pickle

from .InfillCriteria import EI, MGFI
from .optimizer import mies
from .utils import proportional_selection
import os
# TODO: remove the usage of pandas here change it to customized np.ndarray
# TODO: adding logging system

class BayesOpt(object):
    """
    Generic Bayesian optimization algorithm

    A python implementation of a Generic Bayesian optimization algorithm
    that can be used as automatic neural network network configurator using multiple GPU's
    In general it can be used to optimize any (expeonsive) black box problem.

    Args:
        search_space (:obj:`SearchSpace`): The boundaries and dimensions that define the search space.
        obj_func (:obj:`callable`): The objective function, should *print* the fitness value in the end.
        surrogate (:obj:`model`): A sklearn model that is used as surrogate, for example RandomForest, or GaussianProcess
              Any model works as long as it has a train and fit function.
        minimize (boolean, optional): If the objective function is a minimization problem or not, defaults to True.
        noisy (boolean, optional): If the objective function is noizy or not, defaults to False.
        eval_budget (int, optional): The number of objective function evaluations to spend, defaults to None, meaning unlimited.
        max_iter (int, optional): The number of optimization iterations to perform. Default is None, unlimited.
        n_init_sample (int, optional): The number of points for the initial design of experiments. Default is 20 times the number of dimensions.
        n_point (int, optional): Number of candidates to evaluate in paralel. Default is 1.
        n_jobs (int, optional): Number of processes to use for the evaluation. Default is 1.
        n_restart (int, optional): Number of restarts for the optimization over the surrogate model. Default is 10 times the number of dimensions.
        optimizer (str, optional): Internal optimizer, can be 'MIES' or 'BFGS', default is 'MIES'.
        wait_iter (str, optional): Maximal restarts when optimal value does not change, default is 3.
        verbose (boolean, optional): If status info is printed to the console, default is False.
        random_seed (int, optional): Seed of the random generator, default is None for not setting a seed.
        debug (boolean, optional): If additional debug info is printed to the console, default is False.
        resume_file (str, optional): File location for intermediate saves, stores the surrogate in this file and can resume an experiment when the process is killed.
            Defaults to empty string for not using an intermediate file.

    """
    def __init__(self, search_space, obj_func, surrogate,
                 minimize=True, noisy=False, eval_budget=None, max_iter=None,
                 n_init_sample=None, n_point=1, n_jobs=1,
                 n_restart=None, optimizer='MIES', wait_iter=3,
                 verbose=False, random_seed=None,  debug=False, resume_file = ""):

        self.debug = debug
        self.verbose = verbose
        self._space = search_space
        self.var_names = self._space.var_name.tolist()
        self.obj_func = obj_func
        self.noisy = noisy
        self.surrogate = surrogate

        self.resume_file = resume_file
        if (os.path.isfile(resume_file)):
            self.data = self._load_object(resume_file) #resume from previous model
            print("Resuming from previously saved surrogate")
            self.fit_and_assess()

        self.n_point = n_point
        self.n_jobs = min(self.n_point, n_jobs)

        self.minimize = minimize
        self.dim = len(self._space)

        # column names for each variable type
        self.con_ = self._space.var_name[self._space.id_C].tolist()   # continuous
        self.cat_ = self._space.var_name[self._space.id_N].tolist()   # categorical
        self.int_ = self._space.var_name[self._space.id_O].tolist()   # integer

        self.param_type = self._space.var_type
        self.N_r = len(self.con_)
        self.N_d = len(self.cat_)
        self.N_i = len(self.int_)

        # parameter: objective evaluation
        self.init_n_eval = 1      # TODO: for noisy objective function, maybe increase the initial evaluations
        self.max_eval = int(eval_budget) if eval_budget else np.inf
        self.max_iter = int(max_iter) if max_iter else np.inf
        self.n_init_sample = self.dim * 20 if n_init_sample is None else int(n_init_sample)
        self.eval_hist = []
        self.eval_hist_id = []
        self.iter_count = 0
        self.eval_count = 0

        # paramter: acquisition function optimziation
        mask = np.nonzero(self._space.C_mask | self._space.O_mask)[0]
        self._bounds = np.array([self._space.bounds[i] for i in mask])             # bounds for continuous and integer variable
        # self._levels = list(self._space.levels.values())
        self._levels = np.array([self._space.bounds[i] for i in self._space.id_N]) # levels for discrete variable
        self._optimizer = optimizer
        self._max_eval = int(5e2 * self.dim)
        self._random_start = int(10 * self.dim) if n_restart is None else n_restart
        self._wait_iter = int(wait_iter)    # maximal restarts when optimal value does not change

        # Intensify: the number of potential configuations compared against the current best
        # self.mu = int(np.ceil(self.n_init_sample / 3))
        self.mu = 3

        # stop criteria
        self.stop_dict = {}
        self.hist_perf = []
        self._check_params()

        # set the random seed
        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(self.random_seed)

        copy_reg.pickle(self._eval, dill.pickles) # for pickling

        # paralellize gpus
        self.init_gpus = True
        self.evaluation_queue = queue.Queue()


    def _save_object(self, obj, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    def _load_object(self, filename):
        with open(filename, 'rb') as inputfile:
            obj = pickle.load(inputfile)
            return obj

    def _get_var(self, data):
        """
        get variables from the dataframe
        """
        var_list = lambda row: [_ for _ in row[self.var_names].values]
        if isinstance(data, pd.DataFrame):
            return [var_list(row) for i, row in data.iterrows()]
        elif isinstance(data, pd.Series):
            return var_list(data)

    def _to_dataframe(self, var, index=0):
        if not hasattr(var[0], '__iter__'):
            var = [var]
        var = np.array(var, dtype=object)
        N = len(var)
        df = pd.DataFrame(np.c_[var, [0] * N, [None] * N],
                          columns=self.var_names + ['n_eval', 'perf'])
        df[self.con_] = df[self.con_].apply(pd.to_numeric)
        df[self.int_] = df[self.int_].apply(lambda c: pd.to_numeric(c, downcast='integer'))
        df.index = list(range(index, index + df.shape[0]))
        return df

    def _compare(self, perf1, perf2):
        if self.minimize:
            return perf1 < perf2
        else:
            return perf1 > perf2

    def _remove_duplicate(self, confs):
        """
        check for the duplicated solutions, as it is not allowed
        for noiseless objective functions
        """
        idx = []
        X = self.data[self.var_names]
        for i, x in confs.iterrows():
            x_ = pd.to_numeric(x[self.con_])
            CON = np.all(np.isclose(X[self.con_].values, x_), axis=1)
            INT = np.all(X[self.int_] == x[self.int_], axis=1)
            CAT = np.all(X[self.cat_] == x[self.cat_], axis=1)
            if not any(CON & INT & CAT):
                idx.append(i)
        return confs.loc[idx]

    def _eval(self, x, gpu, runs=1):

        perf_, n_eval = x.perf, x.n_eval
        # TODO: handle the input type in a better way
        #try:    # for dictionary input
        __ = [self.obj_func(x[self.var_names].to_dict(), gpu_no=gpu) for i in range(runs)]
        #except: # for list input
        #    __ = [self.obj_func(self._get_var(x)) for i in range(runs)]
        perf = np.sum(__)

        x.perf = perf / runs if not perf_ else np.mean((perf_ * n_eval + perf))
        x.n_eval += runs

        self.eval_count += runs
        self.eval_hist += __
        self.eval_hist_id += [x.name] * runs

        return x, runs, __, [x.name] * runs

    def evaluate(self, data, gpu, runs=1):
        """ Evaluate the candidate points and update evaluation info in the dataframe
        """
        if isinstance(data, pd.Series):
            self._eval(data, gpu)

        else:
            print("Cannot evaluate")

                    # res = Parallel(n_jobs=self.n_jobs, verbose=False)(
                    #     delayed(self._eval, check_pickle=False)(row, gpu_no[k % len(gpu_no)]) \
                    #     for k, row in data.iterrows())
                    #
                    # x, runs, hist, hist_id = list(zip(*res))
                    # self.eval_count += sum(runs)
                    # self.eval_hist += list(itertools.chain(*hist))
                    # self.eval_hist_id += list(itertools.chain(*hist_id))
                    # for i, k in enumerate(data.index):
                    #     data.loc[k] = x[i]

    def fit_and_assess(self):
        X, perf = self._get_var(self.data), self.data['perf'].values

        # normalization the response for numerical stability
        # e.g., for MGF-based acquisition function
        perf_min = np.min(perf)
        perf_max = np.max(perf)
        perf_ = (perf - perf_min) / (perf_max - perf_min)

        # fit the surrogate model
        self.surrogate.fit(X, perf_)

        self.is_update = True
        perf_hat = self.surrogate.predict(X)
        self.r2 = r2_score(perf_, perf_hat)

        # TODO: in case r2 is really poor, re-fit the model or transform the input?
        if self.verbose:
            print('Surrogate model r2: {}'.format(self.r2))
        return self.r2

    def select_candidate(self):
        self.is_update = False
        # always generate mu + 1 candidate solutions
        while True:
            confs_, acqui_opts_ = self.arg_max_acquisition()
            confs_ = self._to_dataframe(confs_, self.data.shape[0])
            confs_ = self._remove_duplicate(confs_)

            # if no new design site is found, re-estimate the parameters immediately
            if len(confs_) == 0:
                if not self.is_update:
                    # NB no new data samples?
                    # Duplication are commonly encountered in the 'corner'
                    self.fit_and_assess()
                else:
                    warnings.warn('iteration {}: duplicated solution found \
                                by optimization! New points is taken from random \
                                design'.format(self.iter_count))
                    confs_ = self.sampling(N=1)
                    break
            else:
                break
        return confs_

    def gpuworker(self, q, gpu_no):
        while True:
            print('GPU no. {} is waiting for task'.format(gpu_no))

            confs_ = q.get()
            print(confs_)
            self.evaluate(confs_, gpu_no)
            self.data = self.data.append(confs_)
            self.data.perf = pd.to_numeric(self.data.perf)
            self.eval_count += 1

            perf = np.array(self.data.perf)
            self.incumbent_id = np.nonzero(perf == np.min(perf))[0][0]

            # model re-training
            self.fit_and_assess()
            self.iter_count += 1
            self.hist_perf.append(self.data.loc[self.incumbent_id, 'perf'])

            incumbent = self.data.loc[[self.incumbent_id]]
            #return self._get_var(incumbent)[0], incumbent.perf.values

            q.task_done()

            #print "GPU no. {} is waiting for task on thread {}".format(gpu_no, gpu_no)
            if not self.check_stop():
                confs_ = self.select_candidate()
                q.put(confs_)
            else:
                break

    def run(self):
        # initialize
        self.data = pd.DataFrame()
        samples = self._space.sampling(self.n_init_sample)
        initial_data_samples = self._to_dataframe(samples)

        # occupy queue with initial jobs
        for i in range(self.n_jobs):
            self.evaluation_queue.put(initial_data_samples.iloc[i])

        # launch threads for all GPUs
        for i in range(self.n_jobs):
            t = threading.Thread(target=self.gpuworker, args=(self.evaluation_queue, i,))
            t.setDaemon = True
            t.start()

        # wait for queue to be empty
        self.evaluation_queue.join()
        self.stop_dict['n_eval'] = self.eval_count
        self.stop_dict['n_iter'] = self.iter_count

        incumbent = self.data.loc[[self.incumbent_id]]
        return incumbent, self.stop_dict

    def check_stop(self):
        # TODO: add more stop criteria
        if self.iter_count >= self.max_iter:
            self.stop_dict['max_iter'] = True

        if self.eval_count >= self.max_eval:
            self.stop_dict['max_eval'] = True

        return len(self.stop_dict)

    def _acquisition(self, plugin=None, dx=False):
        if plugin is None:
            # plugin = np.min(self.data.perf) if self.minimize else -np.max(self.data.perf)
            # Note that performance are normalized when building the surrogate
            plugin = 0 if self.minimize else -1

        acquisition_func = EI(self.surrogate, plugin, minimize=self.minimize)
        return functools.partial(acquisition_func, dx=dx)

    def arg_max_acquisition(self, plugin=None):
        """
        Global Optimization on the acqusition function
        """
        if self.verbose:
            print('acquisition function optimziation...')

        obj_func = self._acquisition(plugin, dx=dx)
        candidates, values = self._argmax_multistart(obj_func)

        return candidates, values

    def _argmax_multistart(self, obj_func):
        # keep the list of optima in each restart for future usage
        xopt, fopt = [], []
        eval_budget = self._max_eval
        best = -np.inf
        wait_count = 0

        for iteration in range(self._random_start):
            x0 = self._space.sampling(1)[0]

            opt = mies(x0, obj_func, self._bounds.T, self._levels, self.param_type,
                       eval_budget, minimize=False, verbose=False)
            xopt_, fopt_, stop_dict = opt.optimize()

            if fopt_ > best:
                best = fopt_
                wait_count = 0
                if self.verbose:
                    print('[DEBUG] restart : {} - funcalls : {} - Fopt : {}'.format(iteration + 1,
                        stop_dict['funcalls'], fopt_))
            else:
                wait_count += 1

            eval_budget -= stop_dict['funcalls']
            xopt.append(xopt_)
            fopt.append(fopt_)

            if eval_budget <= 0 or wait_count >= self._wait_iter:
                break
        # maximization: sort the optima in descending order
        idx = np.argsort(fopt)[::-1]
        return xopt[idx[0]], fopt[idx[0]]

    def _check_params(self):
        assert hasattr(self.obj_func, '__call__')

        if np.isinf(self.max_eval) and np.isinf(self.max_iter):
            raise ValueError('max_eval and max_iter cannot be both infinite')
