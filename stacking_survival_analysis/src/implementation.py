import numpy as np
from utils import import_data
from stacking import SurvivalStacking
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Import the data using the import_data() function
data = import_data()

# Extract the training and test data
X_train = data[['feature_1', 'feature_2']].values
y_train = data['event']
X_test = np.array(data.sample(2)[['feature_1', 'feature_2']].values.tolist())

# Verify the shapes of the data
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)

# Create an instance of the SurvivalStacking model
stacking_model = SurvivalStacking(base_models=[GradientBoostingClassifier(), DecisionTreeClassifier()],
                                  meta_model=GradientBoostingClassifier())

# Check if the stacking_model has been fitted
if not hasattr(stacking_model, 'meta_model_'):
    print("Stacking model is not fitted yet.")

# Fit the stacking_model with the training data
stacking_model.fit(X_train, y_train)

# Verify if the stacking_model has been fitted
if hasattr(stacking_model, 'meta_model_'):
    print("Stacking model is now fitted.")

# Make predictions using the fitted model
predictions = stacking_model.predict(X_test)

# Print the predictions
print(predictions)
