# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 18:53:49 2019

@author: Sapiens
"""
import  matplotlib.pyplot as plt
import numpy as np
import itertools

def onehot(y):
    a = np.zeros((y.shape[0], 10))
    a[np.arange(y.shape[0]), y] = 1
    return a


def plot_acc_loss(history):
    plt.subplot(1,2,1)
    plt.grid(True)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.grid(True)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_confusion_matrix(cmatrix, classes, cmap=plt.cm.Blues):
    print(cmatrix)
    plt.style.use('classic')
    plt.imshow(cmatrix, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes)+1)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cmatrix.max() / 2.
    for i, j in itertools.product(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
        plt.text(j, i, format(cmatrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cmatrix[i, j] > thresh else "black", fontsize=15)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.figure(figsize=(15,8))
    



def plot_fp_results(X_test, y_test, y_pred, indeces, classes):
    
    for i in indeces:
        plt.figure()
        plt.imshow(np.reshape(X_test[i], (28, 100)),cmap='gray')
        plt.title('Position: {}; Expected: {}; Predicted {}'.format(i, classes[y_test[i]], classes[y_pred[i]]))
        plt.savefig("value{i}.png".format(i=i))
        
        
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()