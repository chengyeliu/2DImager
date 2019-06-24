import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras import backend as K
from keras.utils import to_categorical
from keras.constraints import Constraint, non_neg

#Customize Constraint to force weight range
class CustomizedConstraint(Constraint):
    def __init__(self, min_value=0.0, max_value=7.5, rate=2.0, axis=0):
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate
        self.axis = axis

    def __call__(self, w):
        norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
        desired = (self.rate * K.clip(norms, self.min_value, self.max_value) +
                   (1 - self.rate) * norms)
        w *= (desired / (K.epsilon() + norms))
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        return w

    def get_config(self):
        return {'axis': self.axis}

#To simulate CNN to obtain weights for both convolutional and fully-connected layers
def CNN_MNIST(x_train, y_train, x_test, y_test,
             conv2d = False,
             dense = False,
             epoch = 100,
             batch_size = 1024
             ):
    
    #Data Reshaping
    x_train = x_train.copy().reshape(*(x_train.shape),1)
    y_train = to_categorical(y_train.copy())
    x_test = x_test.copy().reshape(*(x_test.shape),1)
    y_test = to_categorical(y_test.copy())

    #Model building
    network = models.Sequential()
    
    #convolutional layer
    if conv2d == False: #to simulate a convolutional layer
        network.add(layers.Conv2D(10,
                              kernel_size=(4, 4), strides=1,
                              bias = 0,
                              input_shape=(a,b,1),
                              kernel_constraint = non_neg(),
                              activation='relu',
                              ))
    elif conv2d != False: #to use weights that are encoded in the array
        network.add(layers.Conv2D(10,
                              trainable = False,
                              kernel_initializer = conv2d_weightmap,
                              kernel_size=(4, 4), strides=1,
                              bias = 0,
                              input_shape=(a,b,1),
                              kernel_constraint = CustomizedConstraint(axis = [0,1]),
                              activation='relu',
                              ))
                              
    #max-pooling
    network.add(layers.MaxPooling2D(pool_size=(5,5)))
    
    #flattening
    network.add(layers.Flatten())
    
    #fully connected layer
    network.add(layers.Dense(10, 
                         kernel_constraint = non_neg(),
                         activation='softmax'))
    
    
    network.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
                  
    model_log = network.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epoch,
              verbose=1,
              validation_data=(x_test, y_test))
    
    return network, model_log
    
if __name__ == "__main__":
    #Import MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    #Image reshaping
    a, b= 13,13
    x_train_resize = np.empty((60000, a, b))
    x_test_resize = np.empty((10000, a, b))

    for i in range(x_train.shape[0]):
        x_train_resize[i] = resize(x_train[i], output_shape = (a,b), preserve_range=True).reshape((a,b))

    for i in range(x_test.shape[0]):
        x_test_resize[i] = resize(x_test[i], output_shape = (a,b), preserve_range=True).reshape((a,b))

    x_train = x_train_resize.astype('float32')/255
    x_test = x_test_resize.astype('float32')/255


    #CNN simulation for weights
    #if conv2d or dense is True, the simulation uses experimentally encoded weights, which can be provided upon request.
    network, model_log = CNN_MNIST(x_train, y_train, x_test, y_test,
             conv2d = False,
             dense = False,
             epoch = 100,
             batch_size = 1024)
