from model import Model
import scipy.io as sio
import layers
import load


def keypoint_detection():
    try:
        data = sio.loadmat('data.mat')
    except:
        load.csv()
        data = sio.loadmat('data.mat')

    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']

    # data normalization
    train_x = train_x / 255.0
    train_y = (train_y - 48) / 48.0
    test_x = test_x / 255.0

    train_x, valid_x = train_x[:-400], train_x[-400:]
    train_y, valid_y = train_y[:-400], train_y[-400:]

    model = Model(1, 0.9, 100, 400)

    model.add_layer(layers.FullConnectedLayer(9216, 256, 0, layers.rectify))
    model.add_layer(layers.FullConnectedLayer(256, 100, 0, layers.rectify))
    model.add_layer(layers.FullConnectedLayer(100, 30, 0, layers.rectify))
    model.set_loss_function(layers.EuclideanLoss)

    model.build()
    print 'build model complete'
    model.train_model(train_x, train_y, valid_x, valid_y)
    model.save_test_result(test_x)


if __name__ == '__main__':
    keypoint_detection()
