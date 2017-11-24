# Tutorial http://scikit-learn.org/stable/tutorial/basic/tutorial.html#machine-learning-the-problem-setting
# https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork
# sklearn has the sample datasets like iris and digits
# load them
# glob helps select multiple files using patterns
import glob
import numpy
# helper to load data from PNG image files
import scipy.misc as sm

# our own image test data set
our_own_dataset = []

for image_file_name in glob.glob('2828_my_own_?.png'):
    print("loading ... ", image_file_name)
    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])
    # load image data from png files into an array
    img_array = sm.imread(image_file_name, flatten=True)
    # reshape from 28x28 to list of 784 values, invert values
    img_data = 255.0 - img_array.reshape(784)
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))
    # append label and image data  to test data set
    record = numpy.append(label, img_data)
    print(record)
    our_own_dataset.append(record)
    pass
