# %%
from tensorflow import keras
import tensorflow as tf
import numpy as np



# %%
#划分训练集与测试集
cifar_10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# %%
# 模型建立
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
# %%
#模型编译
cnn_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# %%
#模型训练
cnn_model.fit(train_images, train_labels, epochs=10, batch_size=32)
# %%
test_loss, test_acc = cnn_model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# %%
probability_model = tf.keras.Sequential([cnn_model, 
                                         tf.keras.layers.Softmax()])

# %%
predictions = probability_model.predict(test_images)

# %%
#随机找几个来验证一下
import random
NO=random.randint(0,9999)

# argmax把输出最大概率的元素，得到结果
print("predict label:"+str(np.argmax(predictions[NO])))

print("ture label:"+str(test_labels[NO]))
# %%
