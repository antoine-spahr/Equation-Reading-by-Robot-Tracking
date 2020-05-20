import os
import sys
sys.path.append('../')
import click
import pickle
import ast

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from src.utils.data_utils import get_MNIST

@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--digits', type=str, default=[0,1,2,3,4,5,6,7,8], help='Digits to consider in training. Default: all but the 9.')
@click.option('--save_path', type=click.Path(exists=False), default='Digit_model.pickle', help='Where to save the model.')
@click.option('--n_rotation', type=int, default='2', help='Number rotated version of the data to train on. Default: 2')
@click.option('--layers', type=str, default=[200,100,50,9], help='MLP structure. The input is a flatten 28x28 image. Deafult: (200,100,50,9).')
def main(data_path, digits, save_path, n_rotation, layers):
    """
    Train a MLP model on the rotated MNIST images in the folder 'data_path'.
    """
    digits = ast.literal_eval(digits)
    layers = ast.literal_eval(layers)
    # get data
    print('>>> Extract data.')
    data = get_MNIST(data_path, digits=digits, add_rotation=n_rotation)
    train_data, train_labels, test_data, test_labels = data
    print(f'>>> {train_data.shape[0]} train samples succesfully loaded.')
    print(f'>>> {test_data.shape[0]} test samples succesfully loaded.')

    # train & test
    MLP_structure = ' -> '.join(['784']+[str(i) for i in layers])
    print(f'>>> Fitting a MLP classifier: {MLP_structure}')
    mlp = MLPClassifier(hidden_layer_sizes=layers,
                        activation='relu',
                        solver='adam',
                        batch_size='auto',
                        learning_rate_init=1e-4,
                        max_iter=300,
                        random_state=69,
                        verbose=True)
    mlp.fit(train_data.reshape(-1, 28*28), train_labels)

    # Evaluate model
    print('>>> Evaluate the fitted model.')
    train_pred = mlp.predict(train_data.reshape(-1, 28*28))
    test_pred = mlp.predict(test_data.reshape(-1, 28*28))
    # print performances
    print('>>> Train Accuracy : '+ '\x1B[32m' + f'{accuracy_score(train_labels, train_pred):.2%}' + '\x1B[0m')
    print(f'>>> Test Accuracy : '+ '\x1B[32m' + f'{accuracy_score(test_labels, test_pred):.2%}' + '\x1B[0m')

    # save model
    with open(save_path, 'wb') as fn:
        pickle.dump(mlp, fn)
    print(f'>>> Model succesfully saved at {save_path}')

if __name__ == '__main__':
    main()
