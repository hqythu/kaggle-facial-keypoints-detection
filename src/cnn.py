from model import Model
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
    (train_x, train_y, text_x) = load.csv()
    print 'load complete'
    model = Model(0.1, 100, 10)
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
    model.train_model(train_x, train_y, train_x, train_y)


if __name__ == '__main__':
    keypoint_detection()

