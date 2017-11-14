# ImageClassy
project 3 for comp 551 

Dependency libraries:

```pip install keras```

```pip install sklearn```

```pip install opencv_python```


To generate data files in .npy format (Preprocessing WIP):
 ```python utils data_dir -preprocess```
-preprocess: Image segmentation with OpenCV using GrabCut, removes noisy background (might not be always accurate tho. So this is basically useless)

To run baseline:
```cd baseline```

```python lr_svm path_to_data_file classifier_to_use```
where classifier_to_use is either "LogisticRegression" or "SVM"

To Run Fully Connected Feedforward Neural Network (folder: ffnn)
1. put "train_x.csv" and "train_y.csv" into a subfolder called "data" in this directory
2. Run the ffnn.ipynb

Note: as mentioned in the report, using 200 neurons per layer and 4 hidden layers takes very long to train (>20 hours)
this code is not optimized for GPUs

To run CNN: download dataset into a same directory
```cd cnn```

```python basic_cnn path_to_data_file```

To run DenseNet and ResNet:
1.cd desenet-resnet
2.pick your favourite model (ex. Densenet-100)
3.run train.py




