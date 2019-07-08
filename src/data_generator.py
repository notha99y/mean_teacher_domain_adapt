import os
from keras.preprocessing.image import ImageDataGenerator


def mean_teacher_data_gen(src_gen, tgt_gen, batch_size, target_size, train_or_val='train'):
    '''

    '''
    if train_or_val == 'train':
        print('Training mode: source domain is synthetic, target domain is real')
        src_path = os.path.join(os.getcwd(), 'reduced_data', 'synthetic')
        tgt_path = os.path.join(os.getcwd(), 'reduced_data', 'real')

    elif train_or_val == 'val':
        print()
        src_path = os.path.join(os.getcwd(), 'reduced_data', 'real')
        tgt_path = os.path.join(os.getcwd(), 'reduced_data', 'synthetic')

    src_data_gen = src_gen.flow_from_directory(src_path,
                                               seed=99,
                                               batch_size=batch_size,
                                               target_size=target_size,
                                               class_mode='categorical')

    tgt_data_gen = tgt_gen.flow_from_directory(tgt_path,
                                               seed=99,
                                               batch_size=batch_size,
                                               target_size=target_size,
                                               class_mode='categorical')

    while True:

        src_img, src_labels = src_data_gen.next()
        tgt_img, _ = tgt_data_gen.next()

        yield [src_img, tgt_img], src_labels


if __name__ == "__main__":
    pass
