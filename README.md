# SurvivalStacking: An API approach to the Stacking-based Survival Analysis Classifier

Chenyang Zhong and Robert Tibshirani's research combines survival analysis with machine learning techniques. They explore the potential of using machine learning algorithms to analyze time-to-event data, typically the survival analysis domain.

In their work, Zhong and Tibshirani propose framing survival analysis as a classification problem. They suggest transforming the time-to-event data into a binary outcome, where each observation is labeled as either an event or censored. By converting survival data into a binary result, they can apply classification algorithms, widely used in machine learning, to predict the event's occurrence.

This approach allows for the application of various machine learning algorithms, such as logistic regression or random forests, to analyze survival data. The authors discuss feature selection and model-building strategies, including variable selection methods like LASSO. They also address model performance assessment and hyperparameter optimization using cross-validation techniques.

## API design and Technical approach
When designing the API for this Survival Analysis Classifier, I used Scikit-Learn API as a reference. The goal is to follow industry best practices and standards for an easy-to-use library. For further review of the Scikit-Learn API, review their paper on their API design [here!](https://arxiv.org/abs/1309.0238).


Design patterns are best practices or templates for solving common software design problems. They are not necessarily tied to code but are more about the approach or the design.

Scikit-Learn has been built using specific design patterns that hide its complexity while providing a simple and semantic interface. This makes it robust and very user-friendly, even for newcomers to machine learning. 

Following Scikit-Learn name conventions, this API includes methods like fit() and predict() that pursue the same goal as sklearn but have a different implementation. 

## Usage

1. Install the necessary dependencies by creating the conda environment:

   ```shell
   conda env create -f environment_droplet.yml
   ```
2. Import the library:

    ```python
    from stacking import SurvivalStacking
    ```
3. Create an instance of the 'SurvivalStacking' class, specifying the base models and meta-model:
    ```python
    stacking_model = SurvivalStacking(base_models=[LogisticRegression(), RandomForestClassifier()],
                                  meta_model=LogisticRegression())
    ```
4. Fit the model on the training data:
    ```python
    stacking_model.fit(X_train, y_train)
    ```
5. Make predictions using the fitted model:
    ```python
    predictions = stacking_model.predict(X_test)
    print(predictions)
    ```

To run implementation:
    ```shell
    cd src
    python implementation.py
    ```

## Unit Tests
Unit tests for the SurvivalStacking class are provided in the test_stacking.py file. You can run the tests using a test runner or by executing the script directly:
```shell
python test_stacking.py
```

## Dependencies
* scikit-learn
* numpy

## License

MIT License

Copyright <2023> <COPYRIGHT alejandro alemany>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.







conda env export > environment_droplet.yml