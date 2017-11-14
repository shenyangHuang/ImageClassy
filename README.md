# ImageClassy
project 3 for comp 551 

Dependency libraries:
```pip install keras```
```pip install sklearn```
```pip install opencv_python```


google folder 
https://drive.google.com/drive/folders/0B2T6WXUfvUxoX3daN0hTNVdXVkE?usp=sharing

To generate data files in .npy format (Preprocessing WIP):
 ```python utils *data_dir* *-preprocess*```
-preprocess: Image segmentation with OpenCV using GrabCut, removes noisy background (might not be always accurate tho. So this is basically useless)

To run baseline:
```cd baseline```
```python lr_svm path_to_data_file classifier_to_use```
where classifier_to_use is either "LogisticRegression" or "SVM"

To run CNN: download dataset into a same directory
```cd cnn```
```python basic_cnn path_to_data_file```