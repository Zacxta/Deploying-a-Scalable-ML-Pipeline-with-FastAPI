# Model Card

## Model Details
This model framework was created by Udacity and edited by Zachary Murfin.
Date:
Version: 1.0
Type: Random Forest

## Intended Use
This model is intented to predict whether individuals make more or less than $50k.

## Training Data
The training data consists of a census dataset split containing entries
with labels on whether or not an individual makes more or less than $50k.

## Evaluation Data
The evaluation data is derived from a separate split of the same census dataset.

## Metrics
Model Parameters:
    n_estimators = 100
    random_state = 42
    max_depth = 20

Metrics:
    Precision: 0.7915
    Recall: 0.6136
    F1: 0.6913

## Ethical Considerations
There are no particular ethical considerations for this model and dataset.

## Caveats and Recommendations
Further tuning of this random forest model would be a good next step.