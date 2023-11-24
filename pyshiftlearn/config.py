# Global
## Paths
### The path of the datasets
DATASET_PATH = "../Datasets"
### The path of the results
RESULTS_PATH = "../results"
### The path of the saved models
SAVED_MODELS_PATH = "../saved_models"

## Neural Networks
### Mini-Batch Gradient Decent
BATCH_SIZE = 64
N_STEPS_TO_PRINT = 20
N_EPOCHS = 5

### Adam
ADAM_BETA1, ADAM_BETA2 = 1e-3, 0.999
ADAM_EPS = 1e-7
ADAM_LR = 1e-3

### Early-Stopping
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 1e-6

# fedorAop
## models
### sampling.py
#### Imblearn Models
IMBLEARN_UNDER_SAMPLING_MODELS = [
    "ClusterCentroids",
    "CondensedNearestNeighbour",
    "EditedNearestNeighbours",
    "RepeatedEditedNearestNeighbours",
    "AllKNN",
    "InstanceHardnessThreshold",
    "NearMiss1",
    "NearMiss2",
    "NearMiss3",
    "NeighbourhoodCleaningRule",
    "OneSidedSelection",
    "RandomUnderSampler",
    "TomekLinks",
]

IMBLEARN_OVER_SAMPLING_MODELS = [
    "RandomOverSampler",
    "SMOTE",
    "ADASYN",
    "BorderlineSMOTE1",
    "BorderlineSMOTE2",
    "KMeansSMOTE",
    "SVMSMOTE",
]

IMBLEARN_COMBINED_SAMPLING_MODELS = [
    "SMOTEENN",
    "SMOTETomek",
]

### neural_network.py
#### Focal Loss
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2

#### RNNLayers
RNN_NUM_LAYERS = 2
RNN_DROPOUT_RATE = 0.2

#### MLP
MLP_NUM_LAYERS = 2
MLP_DROPOUT_RATE = 0.8

if __name__ == "__main__":
    pass