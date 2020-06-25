# Developing a Command line Image Classifier Application with Deep-Learning

This project builds a command line application for training a deep neural network on the flower data set, saves the trained model and makes prediction on any given image.
The classifier is built using a pretained classifier such as VGGNet, RESNet, and ALEXNet using the pytorch framework. 


# Prerequisite

For faster training and inference we recomment you used GPUs. Make sure it has cuda support.


# Specifiactions

The project  includes  two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image





# Running the app


##  Training

Train a new network on a data set with train.py:

```
python train.py data_directory
```

Set directory to save checkpoints:

```
python train.py data_dir --save_dir save_directory
```
Choose architecture:

```
python train.py data_dir --arch "vgg13"
```

Set hyperparameters:

```
python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
```

Use GPU for training:

```
python train.py data_dir --gpu
```


## Prediction

Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability

```
python predict.py /path/to/image checkpoint
```

Return top KK most likely classes:

```
python predict.py input checkpoint --top_k 3
```

Use a mapping of categories to real names:

```
python predict.py input checkpoint --category_names cat_to_name.json
```

Use GPU for inference:

```
python predict.py input checkpoint --gpu
```


