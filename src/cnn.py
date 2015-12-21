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
    train_x = train_x / 256.0
    train_y = (train_y - 48) / 48.0
    test_x = test_x / 256.0

    train_x, valid_x = train_x[:-400], train_x[-400:]
    train_y, valid_y = train_y[:-400], train_y[-400:]

    model = Model(0.1, 0.9, 0.01, 100, 20)
    model.add_layer(layers.ReshapeLayer(1, 96, 96))
    model.add_layer(layers.ConvolutionLayer((5, 5), 2, 1, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2))) # 46 * 46 * 4
    model.add_layer(layers.ConvolutionLayer((5, 5), 4, 2, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2))) # 21 * 21 * 8
    model.add_layer(layers.ConvolutionLayer((4, 4), 8, 4, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2))) # 9 * 9 * 16
    model.add_layer(layers.ConvolutionLayer((4, 4), 16, 8, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2))) # 3 * 3 * 32
    model.add_layer(layers.FullConnectedLayer(144, 30))
    model.set_loss_function(layers.EuclideanLoss)
    model.build()
    print 'build model complete'
    model.train_model(train_x, train_y, valid_x, valid_y)
    model.save_test_result(test_x)


if __name__ == '__main__':
    keypoint_detection()
