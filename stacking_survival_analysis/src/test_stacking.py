import unittest

from utils import import_data
from stacking import SurvivalStacking
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

data = import_data()

class TestSurvivalStacking(unittest.TestCase):
    def test_fit_and_predict(self):
        X_train = data[['feature_1', 'feature_2']].values[:4].tolist()
        y_train = data['event'].values[:4].tolist()
        X_test = data.sample(2)[['feature_1', 'feature_2']].values.tolist()

        stacking_model = SurvivalStacking(base_models=[LogisticRegression(), RandomForestClassifier()], meta_model=LogisticRegression())
        stacking_model.fit(X_train, y_train)
        predictions = stacking_model.predict(X_test)

        self.assertEqual(len(predictions), 2)

    def test_set_meta_model(self):
        stacking_model = SurvivalStacking()

        # Set the meta-classification model
        stacking_model.set_meta_model(LogisticRegression())

        # Assert that the meta-model is set correctly
        self.assertIsInstance(stacking_model.meta_model, LogisticRegression)

    def test_set_base_models(self):
        stacking_model = SurvivalStacking()

        # Set the base classification models
        stacking_model.set_base_models([LogisticRegression(), RandomForestClassifier()])

        # Assert that the base models are set correctly
        self.assertIsInstance(stacking_model.base_models, list)
        self.assertIsInstance(stacking_model.base_models[0], LogisticRegression)
        self.assertIsInstance(stacking_model.base_models[1], RandomForestClassifier)

if __name__ == '__main__':
    unittest.main()
