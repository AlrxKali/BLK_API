import unittest
from utils import import_data
from stacking import SurvivalStacking
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

data = import_data()

class TestSurvivalStacking(unittest.TestCase):
    """Unit tests for the SurvivalStacking class."""

    def test_fit_and_predict(self):
        """Test fitting the model and making predictions."""
        X_train = data[['feature_1', 'feature_2']].values[:4].tolist()
        y_train = data['event'].values[:4].tolist()
        X_test = data.sample(2)[['feature_1', 'feature_2']].values.tolist()

        stacking_model = SurvivalStacking(base_models=[GradientBoostingClassifier(), DecisionTreeClassifier()],
                                          meta_model=GradientBoostingClassifier())
        stacking_model.fit(X_train, y_train)
        predictions = stacking_model.predict(X_test)

        self.assertEqual(len(predictions), 2)

    def test_set_meta_model(self):
        """Test setting the meta-classification model."""
        stacking_model = SurvivalStacking()

        # Set the meta-classification model
        stacking_model.set_meta_model(GradientBoostingClassifier())

        # Assert that the meta-model is set correctly
        self.assertIsInstance(stacking_model.meta_model, GradientBoostingClassifier)

    def test_set_base_models(self):
        """Test setting the base classification models."""
        stacking_model = SurvivalStacking()

        # Set the base classification models
        stacking_model.set_base_models([GradientBoostingClassifier(), DecisionTreeClassifier()])

        # Assert that the base models are set correctly
        self.assertIsInstance(stacking_model.base_models, list)
        self.assertIsInstance(stacking_model.base_models[0], GradientBoostingClassifier)
        self.assertIsInstance(stacking_model.base_models[1], DecisionTreeClassifier)

if __name__ == '__main__':
    unittest.main()
