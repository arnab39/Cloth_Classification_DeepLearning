

### Prerequisites

This code is written in Python2.7 and requires Tensorflow 1.2. In addition, you need to install a few more packages to process MSCOCO data set. 
The MSCOCO data set can be downloaded from  http://cocodataset.org/#download .

Store the downloaded data in 'image/' directory.

You also need to download VGGNet19 model in `data/` directory.

A data folder has to be downloaded from https://drive.google.com/drive/folders/10PHUurBmKUa4pLHT6yOKw4k4rg30DYHh?usp=sharing

You need the following python libraries installed in your system to run the code.

numpy
matplotlib
scipy
scikit-image
hickle
Pillow


For feeding the image to the VGGNet, you have to resize the MSCOCO image dataset to the fixed size of 224x224. 
Run command below then resized images will be stored in `image/train2014_resized/` and `image/val2014_resized/` directory.

```bash
$ python resize.py
```

Before training the model, you have to preprocess the MSCOCO caption dataset.
To generate caption dataset and image feature vectors, run command below.

```bash
$ python prepro.py
```

### Train the model 

To train the image captioning model, run command below. 

```bash
$ python train.py
```

The model runs for 20 epochs and the checkpoints are stored in 'model/lstm/' folder.


### Evaluate the model 

To generate captions, visualize attention weights and evaluate the model, run command below.

```bash
$ python evaluate_model.py
```



