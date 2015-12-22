from model import Model
import sklearn.utils
import layers
import load


def mnist():
    (train_x, train_y, test_x, test_y) = load.mnist('mnist.mat')

    sklearn.utils.shuffle(train_x, train_y, random_state=0)
    sklearn.utils.shuffle(test_x, test_y, random_state=0)

    model = Model(0.01, 0.9, 0.0005, 100, 100)
    model.add_layer(layers.ReshapeLayer(1, 28, 28))
    model.add_layer(layers.ConvolutionLayer((3, 3), 8, 1, 0, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2)))  # 13 * 13 * 8
    model.add_layer(layers.ConvolutionLayer((2, 2), 16, 8, 0, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2)))  # 6 * 6 * 16
    model.add_layer(layers.ConvolutionLayer((3, 3), 32, 16, 0, layers.rectify))
    model.add_layer(layers.PoolingLayer((2, 2)))  # 2 * 2 * 32
    model.add_layer(layers.FullConnectedLayer(128, 128, 0, layers.rectify))
    model.add_layer(layers.DropoutLayer(0.5))
    model.add_layer(layers.FullConnectedLayer(128, 10))
    model.add_layer(layers.SoftmaxLayer())
    model.set_loss_function(layers.CrossEntropyLoss)
    model.build()
    print 'build model complete'
    model.train_model(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    mnist()
