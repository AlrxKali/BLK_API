import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class SurvivalStacking(BaseEstimator, ClassifierMixin):
    """Stacking-based survival analysis classifier.

    Parameters:
        base_models : list, optional
            List of base classification models for stacking. Default is None.
        meta_model : object, optional
            Meta-classification model for stacking. Default is None.

    Attributes:
        base_models : list
            List of base classification models.
        meta_model : object
            Meta-classification model.
        meta_model_ : object
            Fitted meta-classification model.

    """

    def __init__(self, base_models=None, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y, base_models=None):
        """Fit the survival stacking model.

        Parameters:
            X : array-like, shape (n_samples, n_features)
                Input features for training.
            y : array-like, shape (n_samples,)
                Binary outcome indicating whether an event occurred or not.
            base_models : list, optional
                List of base classification models for stacking. Default is None.

        Returns:
            self : object

        """
        # Validate input arrays X and y
        X, y = check_X_y(X, y)

        if base_models is not None:
            self.base_models = base_models

        # Fit the base models
        for model in self.base_models:
            model.fit(X, y)

        # Fit the meta model using base models' predictions
        meta_features = self._get_base_model_predictions(X)

        # Initialize the meta model
        self.meta_model_ = clone(self.meta_model)

        # Fit the meta model using base models' predictions
        self.meta_model_.fit(meta_features, y)

        return self

    def predict(self, X):
        """Predict using the survival stacking model.

        Parameters:
            X : array-like, shape (n_samples, n_features)
                Input features for prediction.

        Returns:
            predictions : array-like, shape (n_samples,)
                Predicted binary outcomes.

        Raises:
            NotFittedError : If the model has not been fitted yet.

        """
        # Check if the model has been fitted
        check_is_fitted(self, 'meta_model_')

        # Validate input array X
        X = check_array(X)

        # Get base models' predictions and make ensemble predictions
        meta_features = self._get_base_model_predictions(X)
        predictions = self.meta_model_.predict(meta_features)

        return predictions

    def set_meta_model(self, meta_model):
        """Set the meta-classification model.

        Parameters:
            meta_model : object
                Meta-classification model for stacking.

        """
        self.meta_model = meta_model

    def set_base_models(self, base_models):
        """Set the base classification models.

        Parameters:
            base_models : list
                List of base classification models for stacking.

        """
        self.base_models = base_models

    def _get_base_model_predictions(self, X):
        """Get predictions from base models.

        Parameters:
            X : array-like, shape (n_samples, n_features)
                Input features for prediction.

        Returns:
            meta_features : array-like, shape (n_samples, n_base_models)
                Predicted probabilities from base models.

        """
        # Get predictions from base models
        meta_features = []
        for model in self.base_models:
            predictions = model.predict_proba(X)[:, 1]
            meta_features.append(predictions)

        return np.column_stack(meta_features)
