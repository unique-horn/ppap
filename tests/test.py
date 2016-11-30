
# coding: utf-8

# In[1]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')


# In[50]:

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, merge
from keras.optimizers import SGD, adam
import numpy as np

from keras.utils import np_utils
from keras.datasets import mnist
from ppap.layers.ppconv import PPConv

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=1)
y_train = np_utils.to_categorical(y_train, 10)
X_test = np.expand_dims(X_test, axis=1)
y_test = np_utils.to_categorical(y_test, 10)


# In[60]:

inputs = Input(shape=(1, 28, 28), name="inputs")
ppconv_1 = PPConv(weight_shape=(3, 3), layer_sizes=[10, 10, 10], nb_filters=2)(inputs)
ppconv_1 =  Activation(activation="relu")(ppconv_1)

ppconv_2 = PPConv(weight_shape=(3, 3), layer_sizes=[10, 10, 10], nb_filters=2)(ppconv_1)
ppconv_2 =  Activation(activation="relu")(ppconv_2)

flat = Flatten()(ppconv_2)
dense_1 = Dense(output_dim=10, activation="softmax")(flat)

model = Model(input=inputs, output=[dense_1])
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.summary()
model.fit(x=X_train, y=y_train, nb_epoch=2, verbose=1)

