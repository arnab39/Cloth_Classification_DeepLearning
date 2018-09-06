from zipfile import ZipFile
import numpy as np

'''load your data here'''

class DataLoader(object):
    def __init__(self):
        self.DIR = '../data/'
    # Returns images and labels corresponding for training and testing. Default mode is train.
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = self.DIR + label_filename + '.zip'
        image_zip = self.DIR + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels

    def create_batches(self,input,output,batch_size):
        perm = np.random.permutation(len(input))
        input = np.array(input)[perm]
        output = np.array(output)[perm]
        n_batches = len(input)/batch_size
        temp = 0
        input_batches = []
        output_batches = []
        for i in range(n_batches):
            input_batches.append(input[temp:(batch_size+temp)])
            output_batches.append(output[temp:(batch_size+temp)])
            temp = i*batch_size
        temp += batch_size
        if len(input)>temp:
            input_batches.append(input[temp:])
            output_batches.append(output[temp:])
        return input_batches,output_batches
