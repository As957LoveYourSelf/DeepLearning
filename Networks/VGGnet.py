import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import losses,Sequential,optimizers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
#
# def preprocess(x, y):
#     # [0~1]
#     x = 2*tf.cast(x, dtype=tf.float32) / 255.-1
#     y = tf.cast(y, dtype=tf.int32)
#     return x,y


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# y_train = tf.squeeze(y_train, axis=1)
# y_test = tf.squeeze(y_test, axis=1)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

# train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
# train_db = train_db.shuffle(1000).map(preprocess).batch(128)
#
# test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
# test_db = test_db.map(preprocess).batch(128)
#
# sample = next(iter(train_db))
# print('Sample:',sample[0].shape,sample[1].shape,tf.reduce_min(sample[0]),tf.reduce_max(sample[0]))



vggnet = Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(64,kernel_size=(3,3),padding="same",activation='relu'),
        layers.Conv2D(64,kernel_size=(3,3),padding="same",activation='relu'),
        layers.MaxPool2D(pool_size=(2,2), strides=2,padding='same'),

        layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'),
        layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),

        layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu'),
        layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),

        layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation='relu'),
        layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),

        layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation='relu'),
        layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),

        layers.Flatten(),
        layers.Dense(256,activation='relu'),
        layers.Dense(64,activation='relu'),
        layers.Dense(10)
    ]
)

vggnet.compile(
    optimizer=optimizers.Adam(lr=0.003),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print(vggnet.summary())
# vggnet.fit(x_train,y_train,batch_size = 128,epochs=10)
# vggnet.evaluate(x_test,y_test,batch_size=128)










