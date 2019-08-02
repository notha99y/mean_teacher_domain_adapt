import os
import json

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import LambdaCallback
from keras.preprocessing.image import ImageDataGenerator

from src.data_generator import mean_teacher_data_gen
from src.model import mean_teacher
from src.model import ema

if __name__ == "__main__":

    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = K.tf.InteractiveSession(config=config)
    K.set_learning_phase(1)  # 1 is training phase, 0 is test phase

    with open(os.path.join(os.getcwd(), 'configs', 'model.json')) as f:
        model_config = json.load(f)

    mean_teacher_model, student_model, teacher_model = mean_teacher(
        model_config)

    with open(os.path.join(os.getcwd(), 'configs', 'training.json')) as f:
        training_config = json.load(f)
    batch_size = training_config['batch_size']
    total_samples = 1200
    weights_dir = 'weights/' + mean_teacher_model.name + \
        '-{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(
        filepath=weights_dir, monitor='loss', save_best_only=False, save_weights_only=False, mode='auto', period=1)
    tensorboard = TensorBoard(log_dir=os.path.join(os.getcwd(), 'logs'), histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True,
                              embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    # cycliclr = CyclicLR(
    #     base_lr=1e-4, max_lr=1e-3, step_size=100., mode='triangular2')
    # early = EarlyStopping(monitor='loss', min_delta=0, patience=5,
    #                       verbose=0, mode='auto', baseline=None, restore_best_weights=False)

    ema_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: ema(
        student_model, teacher_model, alpha=0.99))
    callbacks = [checkpoint, tensorboard, ema_callback]

    with open(os.path.join(os.getcwd(), 'configs', 'augmentation.json')) as f:
        aug_config = json.load(f)
    sup_gen = ImageDataGenerator(featurewise_center=aug_config['featurewise_center'],
                                 featurewise_std_normalization=aug_config['featurewise_std_normalization'],
                                 rescale=1./255,
                                 rotation_range=aug_config['rotation_range'],
                                 width_shift_range=aug_config['width_shift_range'],
                                 height_shift_range=aug_config['height_shift_range'],
                                 horizontal_flip=aug_config['horizontal_flip'])
    unsup_gen = ImageDataGenerator(featurewise_center=aug_config['featurewise_center'],
                                   featurewise_std_normalization=aug_config['featurewise_std_normalization'],
                                   rescale=1./255,
                                   rotation_range=aug_config['rotation_range'],
                                   width_shift_range=aug_config['width_shift_range'],
                                   height_shift_range=aug_config['height_shift_range'],
                                   horizontal_flip=aug_config['horizontal_flip'])
    mean_teacher_generator = mean_teacher_data_gen(
        sup_gen, unsup_gen, training_config['batch_size'], training_config['target_size'])

    mean_teacher_model.fit_generator(mean_teacher_generator,
                                     steps_per_epoch=total_samples // batch_size + 1,
                                     epochs=training_config['epochs'],
                                     callbacks=callbacks)
