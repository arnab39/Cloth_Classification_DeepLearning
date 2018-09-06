from zipfile import ZipFile
import numpy as np

'''load your data here'''

class DataLoader(object):
    def __init__(self,batch_size):
        self.DIR = '../data/'
        self.batch_size = batch_size
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = '../data/' + label_filename + '.zip'
        image_zip = '../data/' + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        images =images - np.mean(images, axis=1)[:, np.newaxis]
        images= images/255.
        return images, labels

    def create_batches(self, images_t, labels_t):
        p = np.random.permutation(len(images_t))
        images=images_t[p]
        labels=labels_t[p]
        images=np.reshape(images,(images.shape[0]/self.batch_size,self.batch_size,images.shape[1]))
        labels=np.reshape(labels,(labels.shape[0]/self.batch_size,self.batch_size))
        return images,labels
'''
data = DataLoader()
data.create_batches()
'''