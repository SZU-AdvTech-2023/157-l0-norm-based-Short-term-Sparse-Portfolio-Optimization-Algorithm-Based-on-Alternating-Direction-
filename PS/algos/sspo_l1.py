import json
import math
import os

import pandas as pd

from universal import tools
from universal.algo import Algo
import numpy as np


class SSPO_L1(Algo):
    """ Bay and hold strategy. Buy equal amount of each stock in the beginning and hold them
    forever.  """

    PRICE_TYPE = 'raw'

    # REPLACE_MISSING = True

    def __init__(self, window=5):
        """
        :params b: Portfolio weights at start. Default are uniform.
        """

        self.window = window
        self.lamda = 0.5
        self.gamma = 0.01
        self.eta = 0.005
        self.zeta = 500
        self.rho = 0
        self.max_iter = int(1e4)
        self.ABSTOL = 1e-4
        self.histLen = 0  # yjf.
        super(SSPO_L1, self).__init__(min_history=self.window)

    def init_weights(self, m):
        len_stro = len(m)
        print("len_stro:",len_stro)
        return np.ones(len_stro) / len_stro

    def calDistance(self, vector1, vector2):
        if len(vector1) != len(vector2):
            return -1
        distance = 0
        for i in range(len(vector1)):
            distance += (vector1[i] - vector2[i]) ** 2
        return math.sqrt(distance)

    def sign(self, x):
        if x > 0:
            res = 1
        elif x < 0:
            res = -1
        else:
            res = 0
        return res

    def positive_element(self, x):
        # print(x)
        for i in range(len(x)):
            # print(x[i])
            if x[i] <= 0:
                x[i] = 0
        return x

    def admm(self, m, last_b, fai_t, lamda, gamma, eta, zeta, rho):
        tao = lamda/gamma
        g = np.copy(last_b)
        b = np.copy(last_b)
        nstk = len(last_b)
        I = np.eye(nstk)
        YI = np.ones((nstk, nstk))
        yi = np.ones(nstk)
        prim_res = []
        # print("len b:",len(b))
        # print(b)
        fai_t = np.array(fai_t)
        for iter in range (1, self.max_iter):
            b = np.linalg.solve((tao * I + eta * YI),(tao * g + (eta - rho)* yi - fai_t))
            g = np.sign(b) * np.maximum(np.abs(b) - gamma, 0)
            prim_res_tmp = np.dot(yi.flatten(), b) -1
            rho = rho + eta * prim_res_tmp
            prim_res.append(prim_res_tmp)

            if np.linalg.norm(prim_res_tmp) < self.ABSTOL:
                break


        b_tplus1_hat = zeta * b
        return list(b_tplus1_hat)

    def step(self, x, last_b, history):
        """

        :param x: the last row data of history
        :param last_b:
        :param history:
        :return:
        """

        # calculate return prediction
        self.histLen = history.shape[0]
        relative_p = self.predict(x, history.iloc[-self.window:])
        fai_t = [0 for i in range(history.shape[1])]
        for i in range(history.shape[1]):
            fai_t[i] = -1.1 * math.log(relative_p[i], 2) - 1
        # print(fai_t)
        b = self.admm(history.shape[1], last_b, fai_t, self.lamda, self.gamma, self.eta, self.zeta, self.rho)
        print(b)
        b = tools.simplex_proj(b)
        # print(b)

        return b

    def predict(self, x, history):
        """ Predict returns on next day. """
        result = []
        for i in range(history.shape[1]):
            temp = max(history.iloc[:, i]) / x[i]
            result.append(temp)
        return result

if __name__ == '__main__':
    datasetName = "D:\SZU_homework\portfolios\\universal-portfolios-master\\universal\data\\tse"
    result = tools.quickrun(SSPO_L1(), tools.dataset(datasetName))
    file_path = "D:\SZU_homework\portfolios\\universal-portfolios-master\\algo_result\sspo_l1-tse.xlsx"
    tools.result_saver(result, file_path)
