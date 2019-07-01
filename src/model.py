import keras
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Model
from keras.applications.resnet50 import ResNet50


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

    return student_model, teacher_model


def add_last_few_layers(base_model, num_of_classes):
    '''
    Adds a GAP, FC 1024, and FC output classes layers to a given base model
    '''
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
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
