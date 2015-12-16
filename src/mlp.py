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

    model = Model(0.1, 100, 100)

    model.add_layer(layers.FullConnectedLayer(9216, 100, 0.001, layers.rectify))
    model.add_layer(layers.FullConnectedLayer(100, 30, 0.001, layers.rectify))
    model.set_loss_function(layers.EuclideanLoss)

    model.build()
    print 'build model complete'
    model.train_model(train_x, train_y)
    model.save_test_result(test_x)


if __name__ == '__main__':
    keypoint_detection()
