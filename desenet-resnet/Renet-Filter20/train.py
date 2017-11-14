import sys,os
import resource

import numpy as np
import sklearn.metrics as metrics

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

import resnet

if __name__ == '__main__':
    batch_size = 50
    nb_classes = 40
    nb_epoch = 100

    img_rows, img_cols = 64, 64
    img_channels = 1

    img_dim = (img_channels, img_rows, img_cols)

    model = resnet.ResnetBuilder.build_resnet_50(img_dim, nb_classes)
    print('Model created')
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    print('Model compiled')

    x = resource.load_train_x(t=20)
    y = resource.load_train_y()
    (trainX, testX) = (x[:int(50000 * 0.9)], x[int(50000 * 0.9):])
    (trainY, testY) = (y[:int(50000 * 0.9)], y[int(50000 * 0.9):])
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    mean_image = np.mean(trainX, axis=0)
    trainX -= mean_image
    testX -= mean_image
    trainX /= 128.0
    testX /= 128.0
    trainY = np_utils.to_categorical(trainY, nb_classes)
    testY = np_utils.to_categorical(testY, nb_classes)

    generator = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=5./64,
                                   height_shift_range=5./64,
                                   horizontal_flip=True)
    generator.fit(trainX, seed=0)

    lr_reducer = ReduceLROnPlateau(monitor='val_acc',
                                   factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=1e-5)
    model_checkpoint = ModelCheckpoint('models/epoch{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.hdf5',
                                       monitor='val_acc',
                                       save_best_only=False,
                                       save_weights_only=True,
                                       verbose=0)
    model_best = ModelCheckpoint('best_models/epoch{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.hdf5',
                                 monitor='val_acc',
                                 save_best_only=True,
                                 save_weights_only=False,
                                 verbose=1)
    csv_logger = CSVLogger('trace.csv')
    callbacks=[lr_reducer, model_checkpoint, model_best, csv_logger]

    model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=nb_epoch,
                        callbacks=callbacks,
                        validation_data=(testX, testY),
                        validation_steps=testX.shape[0] // batch_size,
                        verbose=1)

    model.save_weights('resnet-filter20.hdf5')
    with open('model.json', 'w') as json_file:
        json_file.write(model.to_json())
    
