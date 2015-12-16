import numpy as np
import scipy.io as sio


def mnist(filename):
    data = sio.loadmat(filename)
    train_x = data['train_x'].astype(np.float32)
    test_x = data['test_x'].astype(np.float32)

    train_y = np.zeros((10, train_x.shape[1]))
    test_y = np.zeros((10, test_x.shape[1]))

    for i in range(train_x.shape[1]):
        train_y[data['train_y'][0, i], i] = 1
    for i in range(test_x.shape[1]):
        test_y[data['test_y'][0, i], i] = 1
    return (
        np.transpose(train_x),
        np.transpose(train_y),
        np.transpose(test_x),
        np.transpose(test_y)
    )

def csv():
    f = file('training.csv')
    f.readline()
    text = np.loadtxt(f, dtype=np.str, delimiter=",")

    train_x = np.zeros((len(text), 9216))
    train_y = np.zeros((len(text), 30))

    i = 0
    j = 0
    while i < len(text):
        pic_raw = text[i][30].replace(" ", ",")
        pic = np.array( eval('['+pic_raw+']') )
        i = i + 1
        try:
            train_y[j] = np.array( text[i][0:30] )
            train_x[j] = pic
            j = j + 1
        except:
            j = j

    print j

    f.close()
    f = file('test.csv')
    f.readline()
    text = np.loadtxt(f, dtype=np.str, delimiter=",")

    test_x = np.zeros((len(text), 9216))

    for i in range(len(text)):
        pic_raw = text[i][1].replace(' ', ',')
        pic = np.array( eval('['+pic_raw+']') )
        test_x[i] = pic

    f.close()

    sio.savemat('data.mat', {'train_x': train_x[0:j],
                             'train_y': train_y[0:j],
                             'test_x' : test_x} ) 
