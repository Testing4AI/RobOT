from tensorflow import keras
import tensorflow as tf


 
def Lenet5(input_shape=(28, 28, 1)):
    input_tensor = keras.layers.Input(shape=input_shape)
    
    x = keras.layers.Convolution2D(6, (5, 5), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    x = keras.layers.Convolution2D(16, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
    
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(120, activation='relu', name='fc1')(x)
    x = keras.layers.Dense(84, activation='relu', name='fc2')(x)
    x = keras.layers.Dense(10, name='before_softmax')(x)
    x = keras.layers.Activation('softmax', name='redictions')(x)
    
    return keras.models.Model(input_tensor, x)



def ConvNet_1(input_shape=(28,28,1)):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, 3, input_shape=input_shape, activation='relu'))

    model.add(keras.layers.Conv2D(32, 3, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(64,3, activation='relu'))
    model.add(keras.layers.Conv2D(64,3, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model 
  
