# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains data of Titanic with various features and survivals.

Try to use what you have learnt today to predict whether the passenger shall survive or not.

Evaluate your model.
"""
# %%
# 读取数据
import pandas as pd

traindata = pd.read_csv('..//data//train.csv')
testdata = pd.read_csv('..//data//test.csv')
# data = pd.concat([traindata,testdata],axis=0)
df1=traindata.copy()
df2=testdata.copy()
df1.sample(10)
df2.sample(10)

# %%
# 去除无用特征
df1.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df1.info()
df2.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df2.info()
# %%
# 替换/删除空值，这里是删除
print('Is there any NaN in the datasettrain: {}'.format(df1.isnull().values.any()))
df1.dropna(inplace=True)
print('Is there any NaN in the datasettrain: {}'.format(df1.isnull().values.any()))
print('Is there any NaN in the datasettest: {}'.format(df2.isnull().values.any()))
df2.dropna(inplace=True)
print('Is there any NaN in the datasettest: {}'.format(df2.isnull().values.any()))
# %%
# 把categorical数据通过one-hot变成数值型数据
# 很简单，比如sex=[male, female]，变成两个特征,sex_male和sex_female，用0, 1表示
df1.info()
df1 = pd.get_dummies(df1)
df2.info()
df2 = pd.get_dummies(df2)
# %%
# train-test split
X_train = df1.iloc[:, 1:]
y_train= df1.iloc[:, 0]
X_test=df2.iloc[:, 1:]
y_test=df2.iloc[:, 0]
import numpy as np
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print('X_train: {}'.format(np.shape(X_train)))
print('y_train: {}'.format(np.shape(y_train)))
print('X_test: {}'.format(np.shape(X_test)))
print('y_test: {}'.format(np.shape(y_test)))
# %%
# build model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

models = dict()

models['SVM'] = SVC(kernel='rbf')  # SVM这里我们搞个最常用的
models['KNeighbor'] = KNeighborsClassifier(n_neighbors=5)  # n_neighbors表示neighbor个数
models['RandomForest'] = RandomForestClassifier(n_estimators=100)  # n_estimators表示树的个数
models['MultinomialNB'] = MultinomialNB()
# %%
# predict and evaluate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np


def plot_cm(model, y_true, y_pred, name=None):
    """画混淆矩阵
    :param model: 分类模型
    :param y_true: 标签实际值
    :param y_pred: 标签预测值
    :param name: 模型名称
    """
    _, ax = plt.subplots()
    if name is not None:
        ax.set_title(name)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax)
    plt.show()
    return None


def plot_cm_ratio(model, y_true, y_pred, name=None):
    """画混淆矩阵（按占各类型比例）
    :param model: 分类模型
    :param y_true: 标签实际值
    :param y_pred: 标签预测值
    :param name: 模型名称
    """
    _, ax = plt.subplots()
    if name is not None:
        ax.set_title(name)
    cm = confusion_matrix(y_true, y_pred)
    cm_ratio = np.zeros(cm.shape)
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            cm_ratio[i, j] = cm[i, j] / cm[i].sum()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ratio, display_labels=model.classes_)
    disp.plot(ax=ax)
    plt.show()
    return None


def model_perf(model, y_true, y_pred, name=None):
    """返回模型分类准确率，tpr，fpr
    """
    if name is not None:
        print('For model {}: \n'.format(name))
    cm = confusion_matrix(y_true, y_pred)
    for i in range(len(model.classes_)):
        # TODO: Add comments
        tp = cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tn = cm.sum() - tp - fp - fn
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        acc = (tp + tn) / cm.sum()
        print('For class {}: \n TPR is {}; \n FPR is {}; \n ACC is {}. \n'
        .format(model.classes_[i], tpr, fpr, acc))
    return None


def ovo_eval(model, name=None):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    plot_cm(model, y_test, prediction, name)
    plot_cm_ratio(model, y_test, prediction, name)
    model_perf(model, y_test, prediction, name)
    print('Overall Accuracy: {}'.format(model.score(X_test, y_test)))
# %%
# 评估各模型性能
for name, model in models.items():
    ovo_eval(model, name)
# %%