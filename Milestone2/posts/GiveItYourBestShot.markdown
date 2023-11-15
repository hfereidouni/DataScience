# Part 6: Give it your best shot!

## Model 1: Decision Tree
#### First we have load our code with all features from part 4, then like in Part 5 we have applied one-hot encoding for String features.
### Step 1: Data Pre-processing
1. Drop `change_shot_angle` column where the majority of datas are NaN, then drop NaN values.
2. Using Variance filter method to filter low variance features using `VarianceThreshold`
### Step 2: Hyperparameters tuning
Hyperparameters to tune:
```'class_weight':[{0:1,1:1},{0:3,1:1},{0:6,1:1},{0:12,1:1},{0:50,1:1},{0:100,1:1},{0:1000,1:1}],
    'criterion':['gini','entropy'],
    'max_depth':[1,5,10,15,20,40],
    'min_samples_split':[2,8,16,32,64],
    'min_samples_leaf':[1,2,4,16,32,64]
```
As we can see there are lots of possible combinations, so we have chosen **Randomized Search** to optimize our Algorithme.
### Step 3: Feature selection
After applying wrapper method and optimizing algorithm, we apply a Wrapper method(RFE) to then reduce the half of the features to 6.