from imports import *
from info.info import Information
from dataloader.loader import PreProcessing
from processor.processor import Preprocessor
from model.model import ML


class HouseObjectOriented:
    """
    param train: train data will be used for modelling
    param test:  test data will be used for model evaluation
    """

    def __init__(self):
        # properties
        self.ntrain = None
        self.testID = None
        self.y_train = None
        self.train = None
        self.test = None
        self._info = Information()
        self._Preprocessor = Preprocessor()

        print()
        print('HouseObjectOriented object is created')
        print()

    def add_data(self, train, test):
        # properties
        self.ntrain = train.shape[0]
        self.testID = test.reset_index().drop('index', axis=1)['Id']
        self.y_train = train['price'].apply(lambda x: np.log1p(x))
        self.train = train.drop('price', axis=1)
        self.test = test

        # concatinating the whole data
        self.data = self.concat_data(self.train, self.test)
        self.orig_data = self.data.copy()
        print()
        print('Your data has been added')
        print()

    def concat_data(self, train, test):
        data = pd.concat([self.train.set_index('Id'), self.test.set_index('Id')]).reset_index(drop=True)


        return data

    # using the objects
    def information(self):
        """
        using _info object gives summary about dataset
        :return:
        """
        print(self._info._info_(self.data))

    def preprocessing(self):
        """
        preprocess the data before applying Ml algorithms
        """
        self.data = self._Preprocessor._process(self.data, self.ntrain)

        print()
        print('Data has been Pre-Processed')
        print()

    class visualizer:

        def __init__(self, House_Price_OOP):
            self.hp = House_Price_OOP
            self.data = self.hp.data
            self.ytrain = self.hp.y_train
            self.ntrain = self.hp.ntrain
            self.testID = self.hp.testID
            self.data_vis = data_visualization

        def box_plot(self, columns):
            self.data_vis.box_plot(columns)

        def box_plot(self, columns):
            self.data_vis.box_plot(columns)

        def bar_plot(self, columns):
            self.data_vis.bar_plot(columns)

    class ml:

        def __init__(self, House_Price_OOP):
            self.hp = House_Price_OOP
            self.data = self.hp.data
            self.ytrain = self.hp.y_train
            self.ntrain = self.hp.ntrain
            self.testID = self.hp.testID
            self._ML_ = ML(data=self.data, ytrain=self.ytrain,
                           testID=self.testID, test_size=0.2, ntrain=self.ntrain)

        def show_available_algorithms(self):
            self._ML_.show_available()

        def init_regressors(self, num_models='all'):
            self._ML_.init_ml_regressors(num_models)

        def train_test_validation(self, show_results=True):
            self._ML_.train_test_eval_show_results(show=show_results)

        def cross_validation(self, num_models=4, n_folds=5, show_results=False):
            self._ML_.cv_eval_show_results(num_models=num_models, n_folds=n_folds, show=show_results)

        def visualize_trai_test(self, metrics=['r_squared', 'adjusted r_squared', 'mae', 'mse', 'rmse']):
            self._ML_.visualize_results(cv_train_test='train test', metrics=metrics)

        def visualize_cv(self, metrics=['r_squared', 'rmse']):
            self._ML_.visualize_results(cv_train_test='cv', metrics_cv=metrics)

        def fit_best_model(self):
            self._ML_.fit_best_model()

        def show_predictions(self):
            return self._ML_.show_predictions()

        def save_predictions(self, file_name):
            self._ML_.save_predictions(file_name)
            print('The prediction is saved')
