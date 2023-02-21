from SLP_implementation import CustomPerceptron as slp
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import scipy
import matplotlib.pyplot as plt
import operations
import ast
from scipy.interpolate import make_interp_spline
from scipy.fft import fft, ifft
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split


def append_DF(C1, C2):
    # df = preprocessing()
    # print(C1)
    c1_c2 = C1.append(C2, ignore_index=True, verify_integrity=True, sort=True)

    # print(c1_c2)

    if c1_c2.iloc[0, 5] == 'Adelie':
        c1_c2["species"] = [1 if kind == 'Adelie' else -1 for kind in c1_c2['species']]
    else:
        c1_c2["species"] = [1 if kind == 'Gentoo' else -1 for kind in c1_c2['species']]
    return c1_c2


def split_features(DF, X1, X2, lr, epoch, check, threshold, Y=5):
    # print("DF")
    # print(DF)
    # print("X1")
    # print(X1)
    # print("X2")
    # print(X2)
    X1 = DF.iloc[:, X1];
    X1 = X1.to_numpy()
    X2 = DF.iloc[:, X2];
    X2 = X2.to_numpy()
    Y = DF.iloc[:, Y];
    Y = Y.to_numpy()

    ########## feature 1 ###########
    X1_train_c1 = X1[:30];
    X1_train_c2 = X1[50:80]  # training data 60
    X1_test_c1 = X1[30:50];
    X1_test_c2 = X1[80:100]  # testing data 40
    ########## feature 2 ###########
    X2_train_c1 = X2[:30];
    X2_train_c2 = X2[50:80]  # training data 60
    X2_test_c1 = X2[30:50];
    X2_test_c2 = X2[80:100]  # testing data 40
    ########## Y label ###########
    Y_train_c1 = Y[:30];
    Y_train_c2 = Y[50:80]  # training data 60
    Y_test_c1 = Y[30:50];
    Y_test_c2 = Y[80:100]  # testing data 40
    ''' Appending the X_Train data of feature 1'''
    X1_train = np.append(X1_train_c1, X1_train_c2)
    ''' Appending the X_Train data of feature 2'''
    X2_train = np.append(X2_train_c1, X2_train_c2)
    ''' Appending the X_Test data of feature 1'''
    X1_test = np.append(X1_test_c1, X1_test_c2)
    ''' Appending the X_Test data of feature 2'''
    X2_test = np.append(X2_test_c1, X2_test_c2)
    ''' Appending the Y_Train data'''
    Y_train = np.append(Y_train_c1, Y_train_c2)
    ''' Appending the Y_Test data'''
    Y_test = np.append(Y_test_c1, Y_test_c2)
    ''' Appending the X data of feature 1'''
    X1_all = np.append(X1_train, X1_test)
    ''' Appending the X data of feature 2'''
    X2_all = np.append(X2_train, X2_test)
    ''' Appending the Y data'''
    Y_all = np.append(Y_train, Y_test)
    all = zip(X1_all, X2_all, Y_all)
    df = pd.DataFrame(all, columns=["feature 1", "feature 2", "label"])
    # XTrain, XTest, YTrain,YTest = train_test_splitc(df)
    X = df.iloc[:, df.columns != 'label']
    Y = df['label']
    # print("All")
    # print(all)
    # print("DF")
    # print(df)
    # print ("X")
    # print(X)
    # print("Y")
    # print(Y)
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.4, shuffle=False, random_state=8)
    prcptrn = slp(n_iterations=epoch, learning_rate=lr)
    prcptrn.fit(XTrain, YTrain, check,threshold)
    Xi = np.array(XTrain)
    Xtst = np.array(XTest)
    feat1 = Xtst[:, 0]
    feat2 = Xtst[:, 1]
    max1 = np.max(feat1)
    max2 = np.max(feat2)
    min1 = np.min(feat1)
    min2 = np.min(feat2)
    YTst = np.array(YTest)
    pre = []
    feat1_list = []
    feat2_list = []
    for i in range(XTest.shape[0]):
        pre.append(prcptrn.predict(Xi[i]))
        feat1_list.append((2 * (feat1[i] - min1) / (max1 - min1) - 1))
        feat2_list.append((2 * (feat2[i] - min2) / (max2 - min2) - 1))
    # plt.scatter(Xi[:, 0], Xi[:, 1], color='g')
    # plt.plot(XTst, pre, color='k')
    # plt.show()
    sns.scatterplot(x=feat1_list, y=feat2_list)
    sns.regplot(x=pre, y=YTst)
    plt.show()
    return prcptrn.confusion_matrix(XTest, YTest), prcptrn.score(XTest, YTest)


def visualization(X1, X2, DF):
    sns.scatterplot(x=X1, y=X2, data=DF, hue='species'
                    , palette={0: 'red', 1: 'green', 2: 'blue'})
    plt.show()


# def perceptron_train(in_data,labels,alpha=0.1):
#     errors = []
#     X=np.array(in_data)
#     y=np.array(labels)
#     rgen = np.random.RandomState(1)
#     weights= rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
#     original=weights
#     bias=np.random.random_sample()
#     for key in range(X.shape[0]):
#         #a=activation_func(np.matmul(np.transpose(weights),X[key]))
#         yp=activation_func(np.dot(weights,X[key]) + bias)
#         weights=weights+alpha*(y[key]-yp)* float(X[key])
#         bias=bias+alpha*(y[key]-yp)* 1
#         update = alpha * (y[key] - yp)
#         errors += int(update != 0.0)
#         print('Iteration '+str(key)+': '+str(weights))
#     print('Difference: '+str(weights-original))
#     return weights, bias
#
# def activation_func(X):
#     if X == 0:
#         return 0
#     else:
#         return X / abs(X)
#
# def perceptron_test(in_data,label_shape,weights, bias):
#     X=np.array(in_data)
#     y=np.zeros(label_shape)
#     for key in range(X.shape[1]):
#         a = activation_func(weights*X[key].sum())
#         y[key]=0
#         if a == 1:
#             y[key]=1
#         elif a == -1:
#             y[key]=0
#     return y
#
# def score(result,labels):
#     difference=result-np.array(labels)
#     correct_ctr=0
#     for elem in range(difference.shape[0]):
#         if difference[elem]==0:
#             correct_ctr+=1
#     score=correct_ctr*100/difference.size
#     print('Score='+str(score))

def train_test_splitc(dataset):
    X = dataset.iloc[:, dataset.columns != 'label']
    Y = dataset['label']
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.4, shuffle=False, random_state=8)
    return XTrain, XTest, YTrain, YTest


def preprocessing():
    df = pd.read_csv('penguins.csv')
    df['gender'] = df['gender'].fillna(0)
    df['gender'] = [1 if kind == 'male' else 0 for kind in df['gender']]
    return df

def graph(first, second):
    plt.xlabel()