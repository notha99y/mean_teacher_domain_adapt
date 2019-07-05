import os
from keras.preprocessing.image import ImageDataGenerator


def mean_teacher_data_gen(sup_gen, unsup_gen, batch_size, target_size):
    '''

    '''

    syn_path = os.path.join(os.getcwd(), 'reduced_data', 'synthetic')
    real_path = os.path.join(os.getcwd(), 'reduced_data', 'real')

    sup_data_gen = sup_gen.flow_from_directory(real_path,
                                               seed=99,
                                               target_size=target_size,
                                               class_mode='categorical')

    unsup_data_gen = unsup_gen.flow_from_directory(syn_path,
                                                   seed=99,
                                                   target_size=target_size,
                                                   class_mode='categorical')

    while True:

        syn_img, syn_labels = sup_data_gen.next()
        real_img, _ = unsup_data_gen.next()

        yield [syn_img, real_img], syn_labels


if __name__ == "__main__":
    pass
