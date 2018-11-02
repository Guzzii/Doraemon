from sklearn import base, preprocessing
from scipy import stats

import logging
import progressbar

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)


class ColumnSelector(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, cols):
        self._cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        if isinstance(self._cols, list):
            return X[self._cols].values

        return X.loc[:, X.columns.str.contains(self._cols)].values


class OneHotReshaper(preprocessing.OneHotEncoder):
    """
    One-hot encoder which creates and expands to a new dimension
    """
    def _reshape(self, X):
        if isinstance(self.categories, int):
            return X.reshape(*X.shape[:-1], -1, self.categories))
        else:
            raise NotImplementedError

    def fit_transform(self, X, y=None):
        _tranformed = super(OneHotReshaper, self).fit_transform(X, y)
        return self._reshape(_tranformed)

    def transform(self, X):
        _tranformed = super(OneHotReshaper, self).transform(X)
        return self._reshape(_tranformed)


class BayesianLR(object):
    """
    Bayesian ridge regression class, assuming Guassian on the prior
    of p(w) and likelihood of p(y|X, w, beta)
    """
    def __init__(self, m, s, beta):
        """
        Parameters
        ----------
        m : numpy.ndarray
            Prior mean on the model weights
        s : numpy.ndarray
            Prior variance on the model weights
        beta : scaler
            variance of Guassian error
        """
        self._m_init = self.check_array_shape(m)
        self._s_init = s
        self._beta = beta

        self._weights = {}
        self._weights['prior'] = stats.multivariate_normal(mean=m, cov=s)
        self._weights['posterior'] = None

    def padding(self, x):
        x = self.check_array_shape(x)
        ones = np.ones(shape=(x.shape[0], 1))
        return np.hstack((ones, x))

    @staticmethod
    def check_array_shape(array):
        if len(array.shape) == 1:
            return array.reshape(-1, 1)

        return array

    def _update_weights(self, X, y):
        X = self.padding(X)
        y = self.check_array_shape(y)

        s_init_inversed = np.linalg.inv(self._s_init)
        self._s_updated = np.linalg.inv(s_init_inversed + 1 / self._beta * X.T.dot(X))
        self._m_updated = self._s_updated.dot(
            s_init_inversed.dot(self._m_init) + 1 / self._beta * X.T.dot(y)
        )

        self._weights['posterior'] = stats.multivariate_normal(
            mean=self._m_updated.flatten(), cov=self._s_updated
        )

    def fit(self, X, y):
        self._update_weights(X, y)

    def _predict(self, obs):
        return (self._m_updated.T.dot(obs),
                self._beta + obs.T.dot(self._s_updated).dot(obs))

    def predict(self, X):
        X = self.padding(X)

        self._posterior = [stats.multivariate_normal(*self._predict(obs_i.T))
                           for obs_i in X]
        return self._posterior


class TFBase(object):
    """
    Tensorflow base class -- abstract from graph building, train, predict
    and evaluate
    """
    def __init__(self, average_weights=False, log_dir='logs',
                 model_dir='models', result_dir='results'):
        self._log_dir = log_dir
        self._model_dir = model_dir
        self._result_dir = result_dir
        self._average_weights = average_weights
        self._bar = progressbar.progressbar

    @abstractmethod
    def build_network(self):
        """
        Need to implement self._loss_op, self._opt_op, self._pred_op
        """
        pass

    def build_graph(self):
        with tf.Graph().as_default() as graph:
            self.build_network()
            self._saver = tf.train.Saver()

            if self._average_weights:
                averager = tf.train.ExponentialMovingAverage(decay=0.995)
                self._average_saver = tf.train.Saver(averager.variables_to_restore())

            self._init = tf.global_variables_initializer()

            return graph

    def fit(self, X: dict, y=None, epochs=10, batch_size=64):
        if not isinstance(X, dict):
            raise TypeError('Expect a dictionary for input X')

        self._input_shape = {}
        for key, val in X.items():
            self._n_sample, self._input_shape[key] = val.shape[0], val.shape[1:]

        self._session = tf.Session(graph=self.build_graph())

        with self._session.as_default():
            self._session.run(self._init)

            for epoch in range(epochs):
                permute = np.random.permutation(self._n_sample)

                losses = []
                for start in self._bar(range(0, self._n_sample, batch_size)):
                    end = min(self._n_sample, start + batch_size)
                    batch_idx = permute[start:end]

                    loss, _ = self._session.run(
                        fetches=[self._loss_op, self._opt_op],
                        feed_dict={getattr(self, key): X[key][batch_idx] for key in X}
                    )
                    losses.append(loss)

                logger.info('Epoch: {} - Train Loss: {:5f}'.format(epoch, np.mean(losses)))

    def predict(self, X, batch_size=128):
        n_sample = X.values().__iter__().__next__().shape[0]

        with self._session.as_default():
            y_pred = []
            for start in self._bar(range(0, n_sample, batch_size)):
                end = min(n_sample, start + batch_size)

                y_pred.append(self._session.run(
                    self._pred_op,
                    feed_dict={getattr(self, key): X[key][start:end] for key in X}
                ))

        return np.concatenate(y_pred, axis=0)
