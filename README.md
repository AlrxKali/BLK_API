# SurvivalStacking: Stacking-based Survival Analysis Classifier

Chenyang Zhong and Robert Tibshirani's research combines survival analysis with machine learning techniques. They explore the potential of using machine learning algorithms to analyze time-to-event data, typically the survival analysis domain.

In their work, Zhong and Tibshirani propose framing survival analysis as a classification problem. They suggest transforming the time-to-event data into a binary outcome, where each observation is labeled as either an event or censored. By converting survival data into a binary result, they can apply classification algorithms, widely used in machine learning, to predict the event's occurrence.

This approach allows for the application of various machine learning algorithms, such as logistic regression or random forests, to analyze survival data. The authors discuss feature selection and model-building strategies, including variable selection methods like LASSO. They also address model performance assessment and hyperparameter optimization using cross-validation techniques.

## API design and Technical approach
When designing the API for this Survival Analysis Classifier, I used Scikit-Learn API as a reference. The goal is to follow industry best practices and standards for an easy-to-use library. For further review of the Scikit-Learn API, review their paper on their API design [here!](https://arxiv.org/abs/1309.0238).


Design patterns are best practices or templates for solving common software design problems. They are not necessarily tied to code but are more about the approach or the design.

Scikit-Learn has been built using specific design patterns that hide its complexity while providing a simple and semantic interface. This makes it robust and very user-friendly, even for newcomers to machine learning. 



## Usage

1. Install the necessary dependencies by creating the conda environment:

   ```shell
   conda env create -f environment.yml



conda env export > environment_droplet.yml