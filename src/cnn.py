from model import Model
import scipy.io as sio
import layers
import load


def test_cnn():
    (train_x, train_y, test_x, test_y) = load.mnist('mnist.mat')
    model = Model(0.1, 100, 100)
    model.add_layer(layers.ReshapeLayer(1, 28, 28))
    model.add_layer(layers.ConvolutionLayer((5, 5), 4, 1, 0.01, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2))) # 12 * 12 * 4
    model.add_layer(layers.ConvolutionLayer((3, 3), 8, 4, 0.01, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2))) # 5 * 5 * 8
    model.add_layer(layers.FullConnectedLayer(200, 10, 0.01))
    model.add_layer(layers.SoftmaxLayer())
    model.set_loss_function(layers.CrossEntropyLoss)
    model.build()
    model.train_model(train_x, train_y, test_x, test_y)

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

    model = Model(0.1, 100, 40)
    model.add_layer(layers.ReshapeLayer(1, 96, 96))
    model.add_layer(layers.ConvolutionLayer((5, 5), 1, 1, 0.01, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2))) # 46 * 46 * 4
    model.add_layer(layers.ConvolutionLayer((5, 5), 1, 1, 0.01, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2))) # 21 * 21 * 8
    # model.add_layer(layers.ConvolutionLayer((4, 4), 16, 8, 0.01, layers.rectify))
    # model.add_layer(layers.PoolingLayer((2, 2))) # 9 * 9 * 16
    # model.add_layer(layers.ConvolutionLayer((4, 4), 32, 16, 0.01, layers.rectify))
    # model.add_layer(layers.PoolingLayer((2, 2))) # 3 * 3 * 32
    model.add_layer(layers.FullConnectedLayer(441, 30, 0.01))
    model.set_loss_function(layers.EuclideanLoss)
    model.build()
    print 'build model complete'
    model.train_model(train_x, train_y)
    model.save_test_result(test_x)


if __name__ == '__main__':
    keypoint_detection()
