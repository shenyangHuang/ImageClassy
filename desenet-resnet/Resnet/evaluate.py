from keras.models import model_from_json
from keras.optimizers import Adam
from keras import backend as K
import resource
import numpy

import resnet

batch_size = 50
nb_classes = 40
nb_epoch = 100

img_rows, img_cols = 64, 64
img_channels = 1

img_dim = (img_channels, img_rows, img_cols)

model = resnet.ResnetBuilder.build_resnet_50(img_dim, nb_classes)

model.load_weights('best_models/epoch72-0.96-0.34.hdf5', by_name=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-3),
              metrics=['accuracy'])

model.summary()

testX = resource.load_test_x(t=255).astype('float32')

mean_image = numpy.mean(testX, axis=0)
testX -= mean_image
testX /= 128.0

yPreds = model.predict(testX, verbose=1)
yPreds = numpy.argmax(yPreds, axis=1).flatten()

m = resource.load_reverse_category_map()
map_func = numpy.vectorize(lambda x: m[int(x)])
yPreds = numpy.uint8(map_func(yPreds))

result = numpy.int32(numpy.zeros((yPreds.shape[0], 2)))
for i in range(yPreds.shape[0]):
	result[i,0] = i + 1
	result[i,1] = yPreds[i]
numpy.savetxt('result.csv', result, fmt='%d', delimiter=',', header='Id,Label', comments='')
