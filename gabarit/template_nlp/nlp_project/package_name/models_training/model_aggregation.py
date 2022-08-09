#!/usr/bin/env python3

## Model Agrégation

# Copyright (C) <2018-2022>  <Agence Data Services, DSI Pôle Emploi>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Classes :
# - ModelAgregation -> model aggregation with ModelClass

import os
import json
import pickle
import logging

import numpy as np
import pandas as pd
from typing import List, Callable
from {{package_name}} import utils
from {{package_name}}.models_training import utils_models
from {{package_name}}.monitoring.model_logger import ModelLogger
from {{package_name}}.models_training.model_class import ModelClass

class ModelAggregation(ModelClass):
    '''Model for aggregating multiple ModelClasses'''
    _default_name = 'model_aggregation'

    def __init__(self, list_models: List = None, aggregation_function: str = 'majority_vote', using_proba: bool = None, **kwargs) -> None:
        '''Initialization of the class (see ModelClass for more arguments)

        Args:
            list_models (list) : list of model to be aggregated
            aggregation_function (Callable or str) : aggregation function used
            using_proba (bool) : which object is being aggregated (the probas or the predictions).
                                useless if aggregation_function is a str
        Raises:
            TypeError : if the aggregation_function object is not of type str or Callable
            ValueError : if the object aggregation_function is a str but not found in the dictionary dict_aggregation_function
            ValueError : if the object aggregation_function is not adapte the value using_proba
            ValueError : if use multi-labels
        '''
        # Init.
        super().__init__(**kwargs)

        if self.multi_label:
            raise ValueError("Model aggregation does not support multi-labels")

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        #Get the aggregation function
        dict_aggregation_function = {'majority_vote': {'function': self.majority_vote, 'using_proba': False},
                                    'proba_argmax': {'function': self.proba_argmax, 'using_proba': True}}
        if not isinstance(aggregation_function,(Callable, str)):
            raise TypeError('The aggregation_function objects must be of the callable or str types.')
        if isinstance(aggregation_function,str):
            if aggregation_function not in dict_aggregation_function.keys():
                raise ValueError(f"The aggregation_function object ({aggregation_function}) is not a valid option ({dict_aggregation_function.keys()})")
            if using_proba is None:
                self.using_proba = dict_aggregation_function[aggregation_function]['using_proba']
            elif using_proba != dict_aggregation_function[aggregation_function]['using_proba']:
                raise ValueError(f"The aggregation_function object ({aggregation_function}) is not compatible with using_proba=({self.using_proba})")
            else:
                self.using_proba = using_proba

        # Manage model
        self.aggregation_function = aggregation_function
        self.list_models = list_models
        self.list_real_models = None
        self.list_models_names = None
        if list_models is not None:
            self._get_real_models()
        print('init-----------------------------')

    def _get_real_models(self) -> None:
        '''Populate the self.list_real_models if it is None. Also transforms the ModelClass in self.list_models to the corresponding str if need be.
        '''
        if self.list_real_models is None:
            list_real_models = []
            #Get the real model or keep it
            for model in self.list_models:
                if isinstance(model,str):
                    real_model, _ = utils_models.load_model(model)
                else:
                    real_model = model
                list_real_models.append(real_model)
            self.list_real_models = list_real_models

    def fit(self, x_train, y_train, **kwargs) -> None:
        '''Trains the model
           **kwargs enables Keras model compatibility.

        Args:
            x_train (?): Array-like, shape = [n_samples]
            y_train (?): Array-like, shape = [n_samples]
        Raises:
            RuntimeError: If the model is already fitted
        '''
        if self.trained:
            self.logger.error("The model cannot be refitted.")
            self.logger.error("fit with the new model")
            raise RuntimeError("The model cannot be refitted.")

        self._get_real_models()

        # Fit each model
        list_models = []
        for model in self.list_real_models:
            model.fit(x_train, y_train, **kwargs)

        #Set trained
        self.trained = True
        self.nb_fit += 1

        # Set list_classes based on the list_classes of the first model
        self.list_classes = self.list_real_models[0].list_classes.copy()
        # Set dict_classes based on list classes
        self.dict_classes = {i: col for i, col in enumerate(self.list_classes)}

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict(self, x_test, return_proba:bool = None, **kwargs) -> np.array:
        '''Prediction

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (np.array): array of shape = [n_samples]
        '''
        return_proba = self.using_proba if return_proba is None else return_proba

        # We decide whether to rely on each model's probas or their prediction
        if return_proba:
            probas = self._get_probas(x_test,**kwargs)
            return self.aggregation_function(probas)
        else:
            dict_predict = self._get_predictions(x_test,**kwargs)
            df = pd.DataFrame(dict_predict)
            # self.aggregation_function is the function that actually does the aggregation work
            df['prediction_finale'] = df.apply(lambda x:self.aggregation_function(x), axis=1)
            return df['prediction_finale'].values

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def _get_probas(self, x_test, **kwargs) -> list:
        '''Recover the probability of each model being aggregated

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (list): array of shape = [n_samples, n_features]
        Raises:
            AttributeError: if not self.using_proba
        '''
        self._get_real_models()
        # Predict for each model
        list_predict_proba = []
        for model in self.list_real_models:
            list_predict_proba.append(model.predict_proba(x_test,**kwargs))
        return list_predict_proba

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def _get_predictions(self, x_test, **kwargs) -> dict:
        '''Recover the probability of each model being aggregated

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (dict): dictionary in which the values are lists of underlying model predictions
        '''
        self._get_real_models()
        dict_predict = {}
        # Predict for each model
        for i, model in enumerate(self.list_real_models):
            dict_predict[i] = model.predict(x_test,**kwargs)
        return dict_predict

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.array:
        '''Predicts the probabilities on the test set

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (np.array): array of shape = [n_samples, n_classes]
        '''
        list_predict_proba = self._get_probas(x_test,**kwargs)
        # The probas of all models are averaged.
        return sum(list_predict_proba)/len(self.list_models)

    def proba_argmax(self, proba:List) -> np.array:
        '''Aggregation_function: We take the argmax of the mean of the probabilities of the underlying models to provide a prediction

        Args:
            (List): list of the probability of each model
        Returns:
            (np.array): array of shape = [n_samples]
        Raises:
            AttributeError: if not self.using_proba
        '''
        if not self.using_proba:
            raise AttributeError(f"majority_vote is not compatible with using_proba=False")

        proba_average = sum(proba)/len(self.list_models)
        def get_class(x):
            return self.list_classes[x]
        get_class_v = np.vectorize(get_class)
        return get_class_v(np.argmax(proba_average,axis=1))

    def majority_vote(self, predictions:pd.Series) -> pd.Series:
        '''Aggregation_function: A majority voting system of multiple predictions is used.
        In the case of a tie, we use the first model's prediction (even if it is not in the first votes)

        Args:
            (pd.Series) : the Series containing the predictions of the various models
                          series in which the values are the lists of underlying model predictions
        Return:
            (pd.Series) : majority_vote
        Raises:
            AttributeError: if self.using_proba
        '''
        if self.using_proba:
            raise AttributeError(f"majority_vote is not compatible with using_proba=True")
        if self.multi_label == True:
            self.logger.warning("majority_vote is not compatible with the multi-label")

        votes = predictions.value_counts().sort_values(ascending=False)
        if len(votes)>1 and votes.iloc[0]==votes.iloc[1]:
            return predictions[0]
        else:
            return votes.index[0]

    def save(self, json_data: dict = {}) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        Raises:
            ValueError: if the json_data object is not of type dict
        '''
        if type(json_data) is not dict:
            raise ValueError('json_data must be a type dict')

        # Save each model
        models_path = os.path.join(self.model_dir, f'list_models_aggregation')
        if not os.path.exists(models_path):
            os.mkdir(models_path)
        list_models = []
        list_models_names = []
        for i, model in enumerate(self.list_real_models):
            model.model_dir = os.path.join(models_path, model.model_name)
            # model.model_dir = os.path.join(models_path)
            if not os.path.exists(model.model_dir):
                os.mkdir(model.model_dir)
            model.save(json_data=json_data.copy())
            list_models.append(os.path.split(model.model_dir)[-1])
            # list_models.append(model.model_dir)
            list_models_names.append(model.model_name)
        self.list_models = list_models.copy()

        json_data['list_models'] = list_models.copy()
        json_data['list_models_names'] = list_models_names.copy()
        json_data['aggregation_function'] = self.aggregation_function
        json_data['using_proba'] = self.using_proba

        # Save
        list_real_models = self.list_real_models
        self.list_real_models = None
        super().save(json_data=json_data)
        self.list_real_models = list_real_models

    @utils.trained_needed
    def get_and_save_metrics(self, y_true, y_pred, x=None, series_to_add: List[pd.Series] = None, type_data: str = '', model_logger:ModelLogger = None) -> pd.Series:
        '''Function to obtain and save model metrics

        Args:
            y_true (?): array-like, shape = [n_samples, n_features]
            y_pred (?): array-like, shape = [n_samples, n_features]
        Kwargs:
            x (?): array-like or sparse matrix of shape = [n_samples, n_features]
            series_to_add (list): list of pd.Series to add to the dataframe
            type_data (str): dataset type (validation, test, ...)
            model_logger (ModelLogger): custom class to log metrics in ML Flow
        Raises:
            TypeError: if the series_to_add object is not of type list, and has pd.Series type elements
        Returns:
            pd.DataFrame: the df which contains the statistics
        '''
        if series_to_add is not None:
            if sum([1 if type(_) == pd.Series else 0 for _ in series_to_add]) != len(series_to_add):
                raise TypeError("The series_to_add object must be composed of pd.Series")

        for model in self.list_real_models:
            model.get_and_save_metrics(y_true=y_true,
                                        y_pred=y_pred,
                                        x=x,
                                        series_to_add=series_to_add,
                                        type_data=type_data,
                                        model_logger=model_logger)

        return super().get_and_save_metrics(y_true=y_true,
                                            y_pred=y_pred,
                                            x=x,
                                            series_to_add=series_to_add,
                                            type_data=type_data,
                                            model_logger=model_logger)

    def reload_from_standalone(self, model_dir: str, **kwargs) -> None:
        '''Reloads a model aggregation from its configuration and "standalones" files
            Reloads list model from "list_models" files

        Args:
            model_dir (str): Name of the folder containing the model (e.g. model_autres_2019_11_07-13_43_19)
        Kwargs:
            configuration_path (str): Path to configuration file
            model_aggregation (str): Path to standalone model_aggregation
        Raises:
            ValueError: If configuration_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        model_aggregation_path = kwargs.get('model_aggregation_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if model_aggregation_path is None:
            raise ValueError("The argument model_aggregation can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(model_aggregation_path):
            raise FileNotFoundError(f"The file {model_aggregation_path} does not exist")

        # Load confs
        with open(configuration_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        # Can't set int as keys in json, so need to cast it after reloading
        # dict_classes keys are always ints
        if 'dict_classes' in configs.keys():
            configs['dict_classes'] = {int(k): v for k, v in configs['dict_classes'].items()}
        elif 'list_classes' in configs.keys():
            configs['dict_classes'] = {i: col for i, col in enumerate(configs['list_classes'])}

        # Set class vars
        # self.model_name = # Keep the created name
        # self.model_dir = # Keep the created folder
        self.nb_fit = configs.get('nb_fit', 1)  # Consider one unique fit by default
        self.trained = configs.get('trained', True)  # Consider trained by default
        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['x_col', 'y_col',
                          'list_classes', 'dict_classes', 'multi_label', 'level_save',
                          'list_models', 'list_models_names', 'aggregation_function', 'using_proba']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))


        print('---- agg function load list_model:', self.list_models)
        # self._get_real_models()
        print('---- agg function load list_real_model:', self.list_real_models)

        # # Reload
        # with open(model_aggregation_path, 'rb') as f:
        #     self = pickle.load(f)

        # Reload list_real_models
        list_real_models = []
        for model_name in self.list_models_names:
            model_dir = os.path.join(self.model_dir, f'list_models_aggregation', model_name)
            model, _ = utils_models.load_model(model_dir=model_dir, is_path=True)
            list_real_models.append(model)
        self.list_real_models = list_real_models

        # self.list_real_models = '111111111111111111'
        # print('---- agg function load list_real_model:', self.list_real_models)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")