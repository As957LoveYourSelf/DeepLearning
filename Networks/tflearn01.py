import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses,optimizers
from tensorflow.keras.datasets import cifar10
import sys

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

# sys.exit()


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), padding='valid', activation='relu'),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Conv2D(64,(3,3),padding='valid', activation='relu'),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Conv2D(128,(3,3), padding='valid', activation='relu'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(10)
    ]
)

model.compile(
    optimizer=optimizers.Adam(lr=3e-4),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


print(model.summary())
model.fit(x_train,y_train,batch_size = 64,epochs=10)
model.evaluate(x_test,y_test,batch_size=64)
