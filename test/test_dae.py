import torch
import numpy as np
from udlp.autoencoder.denoisingAutoencoder import DenoisingAutoencoder
from utils import readData

if __name__ == "__main__":
    # from lib.Tox21_Data import read
    # x_tr_t, y_tr_t, x_valid_t, y_valid_t, x_te_t, y_te_t = read("./dataset/tox21/", target=0)

    label_name = ['World', 'Sports', 'Business', 'Sci/Tech']
    training_num, valid_num, test_num, vocab_size = 110000, 10000, 7600, 10000
    training_file = 'dataset/agnews_training_110K_10K-TFIDF-words.txt'
    valid_file = 'dataset/agnews_valid_10K_10K-TFIDF-words.txt'
    test_file = 'dataset/agnews_test_7600_10K-TFIDF-words.txt'

    randgen = np.random.RandomState(13)
    trainX, trainY = readData(training_file, training_num, vocab_size, randgen)
    validX, validY = readData(valid_file, valid_num, vocab_size)
    testX, testY = readData(test_file, test_num, vocab_size)

    # preprocess, normalize each dimension to be [0, 1] for cross-entropy loss
    train_max = torch.max(trainX, dim=0, keepdim=True)[0]
    valid_max = torch.max(validX, dim=0, keepdim=True)[0]
    test_max = torch.max(testX, dim=0, keepdim=True)[0]
    print(train_max.size())
    print(valid_max.size())
    print(test_max.size())
    x_max = torch.max(torch.cat((train_max, valid_max, test_max), 0), dim=0, keepdim=True)[0]
    trainX.div_(x_max)
    validX.div_(x_max)
    testX.div_(x_max)

    in_features = trainX.size()[1]
    out_features = 500
    dae = DenoisingAutoencoder(in_features, out_features)
    dae.fit(trainX, validX, lr=1e-3, num_epochs=10, loss_type="cross-entropy")