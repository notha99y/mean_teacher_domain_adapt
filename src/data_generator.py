import os
from keras.preprocessing.image import ImageDataGenerator


def mean_teacher_data_gen(src_gen, tgt_gen, batch_size, target_size, train_or_val='train'):
    '''

    '''
    if train_or_val == 'train':
        print('Training mode: source domain is synthetic, target domain is real')
        src_path = os.path.join(os.getcwd(), 'reduced_data', 'synthetic')
        tgt_path = os.path.join(os.getcwd(), 'reduced_data', 'real')
        seed = None

    elif train_or_val == 'val':
        print('Validation mode: source domain is real, target domain is synthetic')
        src_path = os.path.join(os.getcwd(), 'reduced_data', 'real')
        tgt_path = os.path.join(os.getcwd(), 'reduced_data', 'synthetic')
        seed = 99

    src_data_gen = src_gen.flow_from_directory(src_path,
                                               seed=seed,
                                               batch_size=batch_size,
                                               target_size=target_size,
                                               class_mode='categorical')

    tgt_data_gen = tgt_gen.flow_from_directory(tgt_path,
                                               seed=seed,
                                               batch_size=batch_size,
                                               target_size=target_size,
                                               class_mode='categorical')
    count = 0
    while src_data_gen.n//src_data_gen.batch_size + 1:

        src_img, src_labels = src_data_gen.next()
        tgt_img, _ = tgt_data_gen.next()
        count += src_data_gen.batch_size

        yield [src_img, tgt_img], src_labels


if __name__ == "__main__":
    pass
