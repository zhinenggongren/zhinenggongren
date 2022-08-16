# %% [markdown]

"""
Homework:
The folder '~//data//homework' contains a folder 'Data', containing hand-digits of letters a-z stored in .txt.
Try to establish a network to classify the digits.
`dataLoader.py` offers APIs for loading data.
"""
# %%
import dataLoader as dl
features,labels=dl.readData(r'../data/homework/Data')

# %%
#画个图
import matplotlib.pyplot as plt
no=3002
plt.plot(features[no,0:30],features[no,30:])
plt.title="Real"+labels[no]
plt.show()
print(labels[no])
# %%
# feature engineering (if necessary)
import numpy as np
labelscopy=labels.copy()
for i in range(len(labelscopy)):
    labelscopy[i]=ord(labelscopy[i])-65

labelscopy= np.array(labelscopy)

class_names=['A','B','C','D','E',
             'F','G','H','I','J'
             'K','L','M','N','O'
             'P','Q','R','S','T'
             'U','V','W','X','Y'
             'Z']
             

# %%
# train-test split
train_features=features[0:3000,:]
train_labels=labelscopy[0:3000]
test_features=features[3000:,:]
test_labels=labelscopy[3000:]

import tensorflow as tf
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# %%
# build the network

model = tf.keras.Sequential([
    tf.keras.layers.Dense(60),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26)
])

#%%
# 编译模型
# 优化器选择adam（不知道选啥的时候用adam就完事了）
# 损失函数选用CategoricalCrossentropy
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), #分类损失函数
              metrics=['accuracy'])

# %%
# training
model.fit(train_features, train_labels, epochs=10,batch_size=2)

# %%
# predict and evaluate
test_loss, test_acc = model.evaluate(test_features,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# %%
predictions = model.predict(test_features)

# %%
#随便抽取几个检验
import random
No=random.randint(0,873)

plt.plot(features[3000+No,0:30],features[3000+No,30:])
plt.title="Real"+labels[3000+No]
plt.show()
print("ture label:"+labels[3000+No])
print("predict label:"+chr(65+np.argmax(predictions[No])))



# %%
