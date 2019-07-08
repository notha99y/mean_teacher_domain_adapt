import os
import json
import keras
import keras.backend as K
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50

from keras.callbacks import LambdaCallback

from src.custom_loss import weighted_sum_loss


def mean_teacher(configs):
    '''
    Creates a mean teacher model using keras
    '''

    student_base_model = ResNet50(
        input_shape=configs['input_shape'], include_top=False, weights=None)
    teacher_base_model = ResNet50(
        input_shape=configs['input_shape'], include_top=False, weights=None)

    student_model = add_last_few_layers(
        student_base_model, configs['num_of_classes'])
    teacher_model = add_last_few_layers(
        teacher_base_model, configs['num_of_classes'])

    # ema(student_model, teacher_model, alpha=0.99)

    src_input = student_model.input
    tgt_input = teacher_model.input

    src_output_student = student_model(src_input)
    tgt_output_student = student_model(tgt_input)
    tgt_output_teacher = teacher_model(tgt_input)
    sq_diff_layer = K.sum(K.square(tgt_output_student - tgt_output_teacher))
    model = Model(inputs=[src_input, tgt_input],
                  outputs=src_output_student, name='mean_teacher')

    model.compile(optimizer=Adam(configs['lr']),
                  loss=weighted_sum_loss(
                      sq_diff_layer=sq_diff_layer, ratio=configs['ratio']),
                  metrics=['accuracy'])

    return model, student_model, teacher_model


def add_last_few_layers(base_model, num_of_classes):
    '''
    Adds a GAP, FC 1024, and FC output classes layers to a given base model
    '''
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    output = Dense(num_of_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input,
                  outputs=output, name=base_model.name)
    return model


def ema(student_model, teacher_model, alpha=0.99):
    '''
    Calculates the exponential moving average of the student model weights and updates the teacher model weights\

    formula:
    t_i = alpha * t_{i-1} + (1 - alpha) * s_i, with default alpha = 0.99
    t_i = weights of teacher model in current epoch
    s_i = weights of student model in current epoch

    '''

    student_weights = student_model.get_weights()
    teacher_weights = teacher_model.get_weights()

    assert len(student_weights) == len(teacher_weights), 'length of student and teachers weights are not equal Please check. \n Student: {}, \n Teacher:{}'.format(
        len(student_weights), len(teacher_weights))

    new_layers = []
    for i, layers in enumerate(student_weights):
        new_layer = alpha*(teacher_weights[i]) + (1-alpha)*layers
        new_layers.append(new_layer)
    teacher_model.set_weights(new_layers)


if __name__ == "__main__":
    with open(os.path.join(os.getcwd(), 'configs', 'model.json')) as f:
        model_config = json.load(f)

    mean_teacher_model, student_model, teacher_model = mean_teacher(
        model_config)

    print('mean teacher model', mean_teacher_model.summary())
    print('-'*80)
    print('teacher model', teacher_model.summary())
    print('-'*80)
    print('student model', student_model.summary())
