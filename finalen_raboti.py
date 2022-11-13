import numpy as np
from scipy.misc import imresize
from skimage import io
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.utils import to_categorical
#from keras.models import model_from_json
import funkcii 

#directory = os.listdir('data/')
fonts = [os.path.join('data', d+'/') for d in os.listdir('data/')]
data_links = sorted([os.path.join(f, i) for f in fonts for i in os.listdir(f)])
data_raw = np.array([io.imread(img, as_gray=True).astype(np.float32) for img in data_links])
data = np.array([imresize(img, (28, 126)) for img in data_raw])
data = np.array([img[:,:-26] for img in data])
train = data[:,:,:,np.newaxis] / 255

#1300 nuli, pa 1300 edinici itn...
labels = np.array([np.zeros(1300, dtype=np.int32) + i for i in range(0, 10)]).flatten() 
y = to_categorical(labels)
#y = funkcii.onehot(labels)

train_ratio = 0.60
validation_ratio = 0.20
test_ratio = 0.20
seed = 256
#np.random.shuffle(train)
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 1 - train_ratio, random_state = seed)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                test_size=test_ratio/(test_ratio + validation_ratio),
                                                random_state = seed)



#model = Sequential(name='CNN')
#model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,100,1)))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(10, activation='softmax'))

model = Sequential(name='CNN')
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                                 input_shape=(28,100,1))) # (28, 100, 1)
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

n_epochs = 15
batch_size = 32

csv_logger = CSVLogger('training.csv', append=True)

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=n_epochs,
          verbose=1,
          validation_data=(X_val, y_val),
          callbacks=[csv_logger])
 
#model.save("model.h5")

#model.save_weights("weights.h5")
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)

#load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
#loaded_model.load_weights("weights.h5")   
#model=loaded_model

funkcii.plot_acc_loss(history)


#score = model.evaluate(X_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])


y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, 1)
y_test = np.argmax(y_test, 1)
cmatrix = confusion_matrix(y_test, y_pred)
classes = ['arial', 'bookman', 'calibri', 'century', 'comic', 'dotum', 'georgia', 'gost', 'tahoma', 'times'] 
funkcii.plot_confusion_matrix(cmatrix, classes)


print(classification_report(y_test, y_pred, target_names=classes))


fp_indeces = (~(y_test == y_pred)).nonzero()[0]
funkcii.plot_fp_results(X_test, y_test, y_pred, fp_indeces, classes)


funkcii.print_cm(cmatrix, classes)


#Proverka so custom slika
slika = np.array(io.imread('C:/Users/Sapiens/Desktop/slika_1.png',as_gray=True).astype(np.float32))
slika = np.array(imresize(slika, (28,126)))
slika = np.asarray(slika[:,:-26])
slika = slika[np.newaxis,:,:,np.newaxis] / 255
prediction=model.predict_classes(slika)
print(classes[np.max(prediction)])
