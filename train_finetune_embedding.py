from evaluator_sydney_embedding import Evaluator
#from generator_vgg16 import Generator
from generator_embedding import Generator
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
#from models_finetune import NIC
#from data_manager_finetune import DataManager
from models_embedding import NIC
from data_manager import DataManager
import h5py
import pickle

num_epochs = 500
batch_size = 20
root_path = '../datasets/flicker8_50/'
captions_filename = root_path + 'complete_data.txt'
data_manager = DataManager(data_filename=captions_filename,
                            max_caption_length=30,
                            word_frequency_threshold=2,
                            extract_image_features=False,
                            cnn_extractor='vgg16',
                            image_directory='/home/user2/data_flicker/Flickr8k_Dataset/Flicker8k_Dataset/',
                            split_data=False,
                            dump_path=root_path)

data_manager.preprocess()
print(data_manager.captions[0])
print(data_manager.word_frequencies[0:20])

preprocessed_data_path = root_path 
generator = Generator(data_path=preprocessed_data_path,
                      batch_size=batch_size)

num_training_samples =  generator.training_dataset.shape[0]
num_validation_samples = generator.validation_dataset.shape[0]
print('Number of training samples:', num_training_samples)
print('Number of validation samples:', num_validation_samples)

import numpy as np
import os
GLOVE_DIR = '..'
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

word_index = pickle.load(open(root_path + 'word_to_id.p', 'rb'))
EMBEDDING_DIM = 50
embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIM))
for word, i in word_index.items():
   # if i >= MAX_NB_WORDS:
   #     continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
embedding_matrix[0,0] = 1
embedding_matrix[1,1] = 1
embedding_matrix[2,2] = 1

model = NIC(max_token_length=generator.MAX_TOKEN_LENGTH,
            vocabulary_size=generator.VOCABULARY_SIZE,
            embedding_matrix=embedding_matrix,
            rnn='gru',
            num_image_features=4096,#150528,#generator.IMG_FEATS,
            hidden_size=128,
            embedding_size=50)

'''
# load weight
weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
f = h5py.File(weight_path)
for k in range(f.attrs['nb_layers']):
    print(k)
    if k >= 22:
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[4].layer.layers[k].set_weight(weights)
f.close()
print('model loaded.')
'''

'''
for layer in model.layers[4].layer.layers[:20]:
    layer.trainable = False
'''

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

print(model.summary())
print('Number of parameters:', model.count_params())

training_history_filename = preprocessed_data_path + 'training_history.log'
csv_logger = CSVLogger(training_history_filename, append=False)
model_names = ('../trained_models/flicker8/flicker8_50/' +
               'flicker8_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                         patience=5, verbose=1)

callbacks = [csv_logger, model_checkpoint, reduce_learning_rate]

model.fit_generator(generator=generator.flow(mode='train'),
                    steps_per_epoch=int(num_training_samples / batch_size),
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=generator.flow(mode='validation'),
                    validation_steps=int(num_validation_samples / batch_size))

evaluator = Evaluator(model, data_path=preprocessed_data_path,
                      images_path='/home/user2/data_flicker/Flickr8k_Dataset/Flicker8k_Dataset/')
evaluator.write_captions()
evaluator.display_caption()
