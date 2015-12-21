from model import Model
import scipy.io as sio
import sklearn
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

    sklearn.utils.shuffle(train_x, train_y, random_state=0)

    train_x, valid_x = train_x[:-400], train_x[-400:]
    train_y, valid_y = train_y[:-400], train_y[-400:]

    model = Model(0.01, 0.9, 0.01, 100, 1000)
    model.add_layer(layers.ReshapeLayer(1, 96, 96))
    model.add_layer(layers.ConvolutionLayer((3, 3), 8, 1, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2))) # 47 * 47 * 8
    model.add_layer(layers.ConvolutionLayer((2, 2), 16, 8, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2))) # 23 * 23 * 16
    model.add_layer(layers.ConvolutionLayer((2, 2), 32, 16, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2))) # 11 * 11 * 32
    model.add_layer(layers.ConvolutionLayer((2, 2), 64, 32, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2))) # 5 * 5 * 64
    model.add_layer(layers.ConvolutionLayer((2, 2), 128, 64, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2))) # 2 * 2 * 128
    model.add_layer(layers.FullConnectedLayer(512, 128, layers.rectify))
    model.add_layer(layers.FullConnectedLayer(128, 64, layers.rectify))
    model.add_layer(layers.FullConnectedLayer(64, 30))
    model.set_loss_function(layers.EuclideanLoss)
    model.build()
    print 'build model complete'
    model.train_model(train_x, train_y, valid_x, valid_y)
    model.save_test_result(test_x)


if __name__ == '__main__':
    keypoint_detection()
