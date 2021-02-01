from collections import Counter

import numpy as np
import pandas as pd
from nltk import ngrams
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


class WordRankClassifier:
    """ An example classifier which implements a 1-NN algorithm.
    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self, threshold=300, ngram=3, return_names=False):
        self.ngram = ngram
        self.threshold = threshold
        self.profile_dict = {}
        self.softmax_list = []
        self.fit_features = []
        self.return_names = return_names

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape

        # X, y = check_X_y(X, y)
        # Store the classes seen during fit
        mymap = dict(list(map(reversed, enumerate(unique_labels(y)))))

        self.names = np.unique(y)
        self.X_ = X
        self.y_ = list(map(lambda pred: mymap[pred], y))
        self.classes_ = np.unique(self.y_)
        self.get_features()
        # Return the classifier
        return self

    def get_most_freq(self, word_list):
        most_common = Counter(word_list).most_common(self.threshold)
        return np.array(list(dict(most_common).keys()))

    def get_profile_dict(self):
        for i, class_ in enumerate(self.classes_):
            feats = list(map("".join, ngrams(' '.join(self.X_[self.y_ == class_]), self.ngram)))
            self.profile_dict[class_] = np.array(self.get_most_freq(feats))
        return self.profile_dict

    def load_model(self, path_to_model=None):
        model = pd.read_csv(path_to_model)
        self.profile_dict = model.to_dict()
        self.fit_features = model.to_numpy()

    def save_clf(self):

        pd.DataFrame.from_dict(self.get_profile_dict()).T.to_csv('model.csv', index_label=False, index=False)

    def get_features(self):
        self.fit_features = list(map(
            lambda i: self.get_most_freq(list(map("".join, ngrams(' '.join(self.X_[self.y_ == self.classes_[i]]),
                                                                  self.ngram)))),
            range(len(self.classes_))
        ))
        return self.fit_features

    def _check_similarity(self, document):
        """
            Checks the similarity between the training dataset train_lang and the test_words data_set
        :param document:
        :return:
        """

        doc_feats = np.array(list(map("".join, ngrams(document, self.ngram))))

        similarity_list = map(
            lambda features: np.sum(np.isin(features, doc_feats)),
            self.fit_features
        )
        prediction = np.array(list(similarity_list)).argmax().item()

        return prediction

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        # check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        pred_list = list(map(self._check_similarity, X))

        if self.return_names:
            mymap = dict(enumerate(self.names))
            pred_list = list(map(lambda pred: mymap[pred], pred_list))

        return pred_list
