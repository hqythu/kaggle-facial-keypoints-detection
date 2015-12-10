from model import Model
import layers
import load


def test_mlp():
    (train_x, train_y, test_x, test_y) = load.mnist('mnist.mat')
    model = Model(0.1, 100, 100)

    # Sigmoid Two Hidden Euclidean
    # model.add_layer(layers.FullConnectedLayer(784, 256, 0.1, layers.sigmoid))
    # model.add_layer(layers.FullConnectedLayer(256, 64, 0.1, layers.sigmoid))
    # model.add_layer(layers.FullConnectedLayer(64, 10, 0.1, layers.sigmoid))
    # model.set_loss_function(layers.EuclideanLoss)

    # Relu Two Hidden Euclidean
    # model.add_layer(layers.FullConnectedLayer(784, 256, 0.01, layers.rectify))
    # model.add_layer(layers.FullConnectedLayer(256, 64, 0.01, layers.rectify))
    # model.add_layer(layers.FullConnectedLayer(64, 10, 0.01, layers.rectify))
    # model.set_loss_function(layers.EuclideanLoss)

    # Relu One Hidden Softmax
    # model.add_layer(layers.FullConnectedLayer(784, 256, 0.01, layers.rectify))
    # model.add_layer(layers.FullConnectedLayer(256, 10, 0.01, layers.rectify))
    # model.add_layer(layers.SoftmaxLayer())
    # model.set_loss_function(layers.CrossEntropyLoss)

    # Relu Two Hidden Softmax
    model.add_layer(layers.FullConnectedLayer(784, 256, 0.01, layers.rectify))
    model.add_layer(layers.FullConnectedLayer(256, 64, 0.01, layers.rectify))
    model.add_layer(layers.FullConnectedLayer(64, 10, 0.01, layers.rectify))
    model.add_layer(layers.SoftmaxLayer())
    model.set_loss_function(layers.CrossEntropyLoss)

    model.build()
    model.train_model(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    test_mlp()
