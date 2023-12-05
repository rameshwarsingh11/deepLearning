#Import the required librarires
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#Preprocess the training data set
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    'data/train',
    target_size =(64,64),
    batch_size =32,
    class_mode='binary')


# Preprocess the test data set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary')


#Initializing the CNN
cnn = tf.keras.models.Sequential()

#Enable Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))

# Enable Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

# Add the secon Convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

# Apply Flattening
cnn.add(tf.keras.layers.Flatten())


#Establish full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Add output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))




