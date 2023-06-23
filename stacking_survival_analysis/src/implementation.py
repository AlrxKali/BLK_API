import numpy as np
from utils import import_data
from stacking import SurvivalStacking
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

data = import_data()

X_train = data[['feature_1', 'feature_2']].values
y_train = data['event']
X_test = np.array(data.sample(2)[['feature_1', 'feature_2']].values.tolist())

# Verify data shapes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)

stacking_model = SurvivalStacking(base_models=[LogisticRegression(), RandomForestClassifier()], meta_model=LogisticRegression())

# Check if the stacking_model has been fitted
if not hasattr(stacking_model, 'meta_model_'):
    print("Stacking model is not fitted yet.")

# Call fit() with the data
stacking_model.fit(X_train, y_train)

# Verify if the stacking_model has been fitted
if hasattr(stacking_model, 'meta_model_'):
    print("Stacking model is now fitted.")

predictions = stacking_model.predict(X_test)

print(predictions)