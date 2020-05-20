import os
import sys
sys.path.append('../')
import click
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.utils.data_utils import get_operators_train_data

@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--save_path', type=click.Path(exists=False), default='Operators_model.pickle', help='Where to save the model.')
@click.option('--n_feature', type=int, default='5', help='Number of Fourier descritors to use. Default: 5')
@click.option('--n_neighbors', type=int, default='5', help='Number neighbors to conisder in KNN. Deafult: 5')
def main(data_path, save_path, n_feature, n_neighbors):
    """
    Train a KNN model on the training operators images in the folder 'data_path'.
    """
    # get data
    print('>>> Extract data.')
    data = get_operators_train_data(data_path, Nfeat=n_feature, rotate=True)
    features, labels, label_dict = data
    print(f'>>> {features.shape[0]} samples with {features.shape[1]} features succesfully loaded.')

    # 70-30% split of data
    print('>>> Estimate model performances using a 70-30 train-test split.')
    train_data, test_data, train_labels, test_labels = train_test_split(features, labels, train_size=0.7)

    # train & test
    print(f'>>> Fitting a {n_neighbors}-Nearest-Neighbors classifier.')
    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(train_data, train_labels)

    # Evaluate model
    print('>>> Evaluate the fitted model.')
    train_pred = knn.predict(train_data)
    test_pred = knn.predict(test_data)# print performances
    print('>>> Train Accuracy : '+ '\x1B[32m' + f'{accuracy_score(train_labels, train_pred):.2%}' + '\x1B[0m')
    print(f'>>> Test Accuracy : '+ '\x1B[32m' + f'{accuracy_score(test_labels, test_pred):.2%}' + '\x1B[0m')

    # retrain on all the data
    print('>>> Train the model with all the samples.')
    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(features, labels)

    # save model
    with open(save_path, 'wb') as fn:
        pickle.dump({'model': knn, 'labels_name': label_dict}, fn)
    print(f'>>> Model succesfully saved at {save_path}')

if __name__ == '__main__':
    main()
