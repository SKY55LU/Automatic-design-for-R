"""if possible,
please run the code under the following environment:
python 3.6, pandas 0.20.3, numpy 1.19.5, tensorflow 2.6.2, matplotlib 3.3.4

with support of NBTRM, we can generate the NURBS curve and continues surface metal pattern matrix

through CST-Python co-simulation, the electromagnetic response of the pattern layer can be obtained and made into a dataset

Here is an example, which assumes that you have a dataset folder(csv)
In this file, each line of data is encoded from left to right for a 50-bit pattern, a 20-bit permittivity, and 701-bit of real and 701-bit of imaginary information
The example folder flower_photos should have a structure like this:

~/s11.csv
~/s12.csv
~/s22.csv
~/s13.csv
~/s33.csv"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, LeakyReLU, LayerNormalization,Reshape
from keras.layers import Conv1DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def dtb(num):

    """ The permittivity is converted to 20-bit encoding"""

    if num == int(num):
        integer = '{:04b}'.format(int(num))
        tmpflo = ['0000000000000000']
        return list(map(int, integer+ ''.join(tmpflo)))
    else:
        integer = int(num)
        flo = num - integer
        integercom = '{:04b}'.format(integer)
        tem = flo
        tmpflo = []
        for i in range(16):
            tem *= 2
            tmpflo += str(int(tem))
            tem -= int(tem)
        flocom = tmpflo
        return list(map(int, integercom + ''.join(flocom)))


def scheduler(epoch):

    """ scheduler is used to implement learning-rate dynamic change """

    if epoch < 5:
        return 0.001
    else:
        lr = 0.001 * tf.math.exp(0.1 * (5 - epoch))
        return lr.numpy()


def PILN(featu1, featu2, outpuut1, outpuut2, test_featu1, test_featu2, output1, output2, name):

    """ PILN """

    # input
    input_db = Input(shape=(50, 1), name='fea1')
    input_er = Input(shape=(20, ), name='fea2')

    # initializer
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)

    # three deconvolution
    cov1 = Conv1DTranspose(filters=1, kernel_size=5, strides=2, kernel_initializer=initializer, padding='same')(input_db)
    cov1 = LayerNormalization()(cov1)
    cov1 = LeakyReLU(alpha=0.2)(cov1)
    cov1 = Reshape((100, ))(cov1)

    cov2 = Conv1DTranspose(filters=1, kernel_size=5, strides=2, kernel_initializer=initializer, padding='same')(input_db)
    cov2 = LayerNormalization()(cov2)
    cov2 = LeakyReLU(alpha=0.2)(cov2)
    cov2 = Reshape((100, ))(cov2)

    cov3 = Conv1DTranspose(filters=1, kernel_size=5, strides=2, kernel_initializer=initializer, padding='same')(input_db)
    cov3 = LayerNormalization()(cov3)
    cov3 = LeakyReLU(alpha=0.2)(cov3)
    cov3 = Reshape((100, ))(cov3)

    # the dielectric constant is added in the form of attention
    Dense_s = tf.keras.layers.Concatenate()([cov1, cov2, cov3, input_er])

    # the first dense layer
    Dense_s = Dense(600, kernel_initializer=initializer, bias_initializer='ones')(Dense_s)
    Dense_s = LayerNormalization()(Dense_s)
    Dense_s = LeakyReLU(alpha=0.2)(Dense_s)

    # the second dense layer
    Dense_s = Dense(900, kernel_initializer=initializer, bias_initializer='ones')(Dense_s)
    Dense_s = LayerNormalization()(Dense_s)
    Dense_s = LeakyReLU(alpha=0.2)(Dense_s)

    # the third dense layer
    Dense_s = Dense(1200, kernel_initializer=initializer, bias_initializer='ones')(Dense_s)
    Dense_s = LayerNormalization()(Dense_s)
    Dense_s = LeakyReLU(alpha=0.2)(Dense_s)

    # final layer
    decode_output_a = Dense(701, activation='tanh', kernel_initializer=initializer, bias_initializer='ones', name='fc_a')(Dense_s)
    decode_output_b = Dense(701, activation='tanh', kernel_initializer=initializer, bias_initializer='ones', name='fc_b')(Dense_s)

    auto_encoder = Model(inputs=[input_db, input_er], outputs=[decode_output_a, decode_output_b])

    # optimizer
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.005, amsgrad=False)
    auto_encoder.compile(optimizer=adam, loss='mse', loss_weights={'fc_a': 0.5,'fc_b': 0.5})

    auto_encoder.summary()
    plot_model(auto_encoder, to_file='model.png')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
    auto_encoder.fit(x=[featu1, featu2], y=[outpuut1, outpuut2], validation_data=([test_featu1, test_featu2], [output1, output2]), epochs=100, batch_size=500, shuffle=False,  callbacks=[reduce_lr])

    # save model
    auto_encoder.save('./pre_training/' + name +'.h5')

    # save loss
    loss = auto_encoder.history.history['loss']
    pd.DataFrame(loss).to_csv("./pre_training/loss"+ name + ".csv", mode='a', index=0, header=0)
    vol_loss = auto_encoder.history.history['val_loss']
    pd.DataFrame(vol_loss).to_csv("./pre_training/vol_loss" + name + ".csv", mode='a', index=0, header=0)

    fc_a_loss = auto_encoder.history.history['fc_a_loss']
    pd.DataFrame(fc_a_loss).to_csv("./pre_training/fc_a_loss" + name + ".csv", mode='a', index=0, header=0)
    vol_fc_a_loss = auto_encoder.history.history['val_fc_a_loss']
    pd.DataFrame(vol_fc_a_loss).to_csv("./pre_training/vol_fc_a_loss" + name + ".csv", mode='a', index=0, header=0)

    fc_b_loss = auto_encoder.history.history['fc_b_loss']
    pd.DataFrame(fc_b_loss).to_csv("./pre_training/fc_b_loss" + name + ".csv", mode='a', index=0, header=0)
    vol_fc_b_loss = auto_encoder.history.history['val_fc_b_loss']
    pd.DataFrame(vol_fc_b_loss).to_csv("./pre_training/vol_fc_b_loss" + name + ".csv", mode='a', index=0, header=0)

    # # loss figure show
    # epochs = range(1, len(loss)+1)
    # plt.title('loss')
    # plt.ylim((0, 0.1))
    # plt.plot(epochs, fc_b_loss, 'red', label='Training loss')
    # plt.plot(epochs, vol_fc_b_loss, 'blue', label='validation loss')
    # plt.legend(loc=1)
    # plt.savefig('./pre_training/' + name + '.jpg')
    # # plt.show()
    # plt.close()


def retrain(featu1, featu2, outpuut1, outpuut2, test_featu1, test_featu2, output1, output2, name):

    """ use the pre-training model to retrain"""

    # load model
    auto_encoder = load_model('./pre_training/' + name +'.h5')

    # retrain
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
    auto_encoder.fit(x=[featu1, featu2], y=[outpuut1, outpuut2], validation_data=([test_featu1, test_featu2], [output1, output2]), epochs=100, batch_size=500, shuffle=False,  callbacks=[reduce_lr])

    # save model
    auto_encoder.save('./final_training/' + name + '.h5')

    # save loss
    loss = auto_encoder.history.history['loss']
    pd.DataFrame(loss).to_csv("./final_training/loss" + name + ".csv", mode='a', index=0, header=0)
    vol_loss = auto_encoder.history.history['val_loss']
    pd.DataFrame(loss).to_csv("./final_training/vol_loss" + name + ".csv", mode='a', index=0, header=0)

    fc_a_loss = auto_encoder.history.history['fc_a_loss']
    pd.DataFrame(fc_a_loss).to_csv("./final_training/fc_a_loss" + name + ".csv", mode='a', index=0, header=0)
    vol_fc_a_loss = auto_encoder.history.history['val_fc_a_loss']
    pd.DataFrame(vol_fc_a_loss).to_csv("./final_training/vol_fc_a_loss" + name + ".csv", mode='a', index=0, header=0)

    fc_b_loss = auto_encoder.history.history['fc_b_loss']
    pd.DataFrame(fc_b_loss).to_csv("./final_training/fc_b_loss" + name + ".csv", mode='a', index=0, header=0)
    vol_fc_b_loss = auto_encoder.history.history['val_fc_b_loss']
    pd.DataFrame(vol_fc_b_loss).to_csv("./final_training/vol_fc_b_loss" + name + ".csv", mode='a', index=0, header=0)

    # epochs = range(1, len(loss) + 1)
    # plt.title('loss')
    # plt.ylim((0, 0.1))
    # plt.plot(epochs, loss, 'red', label='Training loss')
    # plt.plot(epochs, vol_loss, 'blue', label='validation loss')
    # plt.legend(loc=1)
    # plt.savefig('./pre_training/' + name + '.jpg')
    # # plt.show()
    # plt.close()

s_list_3 = ['s11', 's21', 's22', 's31', 's33']
for i in range(0, 5, 1):

    """ load data for training"""

    data_total_use = pd.read_csv("./" + s_list_3[i] + ".csv", header=None)

    # Check whether the data is duplicated
    data_total_use_pd = pd.DataFrame(data_total_use)
    drop_list = list(range(0, 70))
    data_total_use_pd.drop_duplicates(subset=drop_list, keep='first', inplace=True)

    # Split the dataset
    data_total_train, data_total_test1 = train_test_split(pd.DataFrame(data_total_use_pd), test_size=0.2, random_state=1)
    data_total_val, data_total_test = train_test_split(pd.DataFrame(data_total_test1), test_size=0.5, random_state=1)
    test_num = data_total_test._stat_axis.values.tolist()

    data_total_train = np.array(data_total_train).astype('float32')
    data_total_val = np.array(data_total_val).astype('float32')
    feature_label_train, feature_er_train, s_train1, s_train2 = np.hsplit(data_total_train, (50, 70, 771))
    feature_label_val, feature_er_val, s_val1, s_val2 = np.hsplit(data_total_val, (50, 70, 771))

    # 训练神经网络
    PILN(feature_label_train, feature_er_train, s_train1, s_train2, feature_label_val, feature_er_val, s_val1, s_val2, s_list_3[i])
    # retrain(feature_label_train, feature_er_train, s_train1, s_train2, feature_label_val, feature_er_val, s_val1, s_val2, s_list_3[i])

