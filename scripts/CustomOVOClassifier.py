from sklearn.multiclass import OneVsOneClassifier
import array
import itertools
import warnings
from numbers import Integral, Real
from pprint import pprint

import numpy as np
import scipy.sparse as sp

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    MultiOutputMixin,
    _fit_context,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.utils._tags import get_tags
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    process_routing,
)
from sklearn.utils.metaestimators import _safe_split, available_if
from sklearn.utils.multiclass import (
    _check_partial_fit_first_call,
    _ovr_decision_function,
    check_classification_targets,
)
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_method_params,
    _num_samples,
    check_is_fitted,
    validate_data,
)


def _fit_binary(estimator, X, y, fit_params, classes=None):
    """Fit a single binary estimator."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn(
                "Label %s is present in all training examples." % str(classes[c])
            )
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator = clone(estimator)
        estimator.fit(X, y, **fit_params)
    return estimator

class _ConstantPredictor(BaseEstimator):
    """Helper predictor to be used when only one class is present."""

    def fit(self, X, y):
        check_params = dict(
            ensure_all_finite=False, dtype=None, ensure_2d=False, accept_sparse=True
        )
        validate_data(
            self, X, y, reset=True, validate_separately=(check_params, check_params)
        )
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self)
        validate_data(
            self,
            X,
            ensure_all_finite=False,
            dtype=None,
            accept_sparse=True,
            ensure_2d=False,
            reset=False,
        )

        return np.repeat(self.y_, _num_samples(X))

    def decision_function(self, X):
        check_is_fitted(self)
        validate_data(
            self,
            X,
            ensure_all_finite=False,
            dtype=None,
            accept_sparse=True,
            ensure_2d=False,
            reset=False,
        )

        return np.repeat(self.y_, _num_samples(X))

    def predict_proba(self, X):
        check_is_fitted(self)
        validate_data(
            self,
            X,
            ensure_all_finite=False,
            dtype=None,
            accept_sparse=True,
            ensure_2d=False,
            reset=False,
        )
        y_ = self.y_.astype(np.float64)
        return np.repeat([np.hstack([1 - y_, y_])], _num_samples(X), axis=0)


def _fit_ovo_binary(estimator, X, y, i, j, fit_params):
    """Fit a single binary estimator (one-vs-one)."""
    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    y_binary = np.empty(y.shape, int)
    y_binary[y == i] = 0
    y_binary[y == j] = 1
    indcond = np.arange(_num_samples(X))[cond]

    fit_params_subset = _check_method_params(X, params=fit_params, indices=indcond)
    return (
        _fit_binary(
            estimator,
            _safe_split(estimator, X, None, indices=indcond)[0],
            y_binary,
            fit_params=fit_params_subset,
            classes=[i, j],
        ),
        indcond,
    )


class CustomOneVsOneClassifier(OneVsOneClassifier):

    def __init__(self, estimators, *, n_jobs=None):
        #Changed to allow classifiers for each pair rather than using the same classifier for all
        self.estimator = estimators
        self.n_jobs = n_jobs


    def fit(self, X, y, **fit_params):
        _raise_for_params(fit_params, self, "fit")

        routed_params = process_routing(
            self,
            "fit",
            **fit_params,
        )

        # We need to validate the data because we do a safe_indexing later.
        X, y = validate_data(
            self, X, y, accept_sparse=["csr", "csc"], ensure_all_finite=False
        )
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError(
                "OneVsOneClassifier can not be fit when only one class is present."
            )
        n_classes = self.classes_.shape[0]
        estimators_indices = list(
            zip(
                *(
                    Parallel(n_jobs=self.n_jobs)(
                        delayed(_fit_ovo_binary)(
                            #Changed to allow classifiers for each pair rather than using the same classifier for all
                            self.estimator[str(i)+"_"+str(j)],
                            X,
                            y,
                            self.classes_[i],
                            self.classes_[j],
                            fit_params=routed_params.estimator.fit,
                        )
                        for i in range(n_classes)
                        for j in range(i + 1, n_classes)
                    )
                )
            )
        )
        
        self.estimators_ = estimators_indices[0]

        pairwise = self.__sklearn_tags__().input_tags.pairwise
        self.pairwise_indices_ = estimators_indices[1] if pairwise else None

        return self