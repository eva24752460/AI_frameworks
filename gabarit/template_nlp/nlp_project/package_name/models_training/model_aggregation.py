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


import logging
import os

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

    def __init__(self, list_models: List, aggregation_function='majority_vote', using_proba:bool=None, **kwargs) -> None:
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
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        self.using_proba = using_proba

        #Get the aggregation function
        dict_aggregation_function = {'majority_vote':{'function': self.majority_vote, 'using_proba': False},
                                    'proba_argmax':{'function': self.proba_argmax, 'using_proba': True}}
        if not isinstance(aggregation_function,(Callable,str)):
            raise TypeError('The aggregation_function objects must be of the callable or str types.')
        if isinstance(aggregation_function,str):
            if aggregation_function not in dict_aggregation_function.keys():
                raise ValueError(f"The aggregation_function object ({aggregation_function}) is not a valid option ({dict_aggregation_function.keys()})")
            if self.using_proba is None:
                self.using_proba = dict_aggregation_function[aggregation_function]['using_proba']
            elif self.using_proba != dict_aggregation_function[aggregation_function]['using_proba']:
                raise ValueError(f"The aggregation_function object ({aggregation_function}) does not support using_proba=({self.using_proba})")
            aggregation_function = dict_aggregation_function[aggregation_function]['function']
        self.aggregation_function = aggregation_function

        # Manage model
        self.list_models = list_models
        self.list_real_models = None
        self._get_real_models()

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

        # Set list_classes based on the list_classes of the first modèle
        self.list_classes = self.list_real_models[0].list_classes.copy()
        # Set dict_classes based on list classes
        self.dict_classes = {i: col for i, col in enumerate(self.list_classes)}

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict(self, x_test, **kwargs) -> np.array:
        '''Prediction

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (np.array): array of shape = [n_samples]
        '''

        # We decide whether to rely on each model's probas or their prediction
        if self.using_proba:
            proba = self._get_probas(x_test,**kwargs)
            return self.aggregation_function(proba)
        else:
            dict_predict = self._get_predictions(x_test,**kwargs)
            df = pd.DataFrame(dict_predict)
            # self.aggregation_function is the function that actually does the aggregation work
            df['prediction_finale'] = df.apply(lambda x:self.aggregation_function(x),axis=1)
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
        if not self.using_proba:
            raise AttributeError(f"The aggregation_function object proba_argmax does not support using_proba=False")

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
        Raises:
            AttributeError: if self.using_proba
        '''
        if self.using_proba:
            raise AttributeError(f"The aggregation_function object proba_argmax does not support using_proba=False")

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
        Raises:
            AttributeError: if not self.using_proba
        '''
        if not self.using_proba:
            raise AttributeError(f"The aggregation_function object proba_argmax does not support using_proba=False")

        list_predict_proba = self._get_probas(x_test,**kwargs)
        # The probas of all models are averaged.
        return sum(list_predict_proba)/len(self.list_models)

    def proba_argmax(self, proba:List) -> np.array:
        '''We take the argmax of the mean of the probabilities of the underlying models to provide a prediction

        Args:
            (List): list of the probability of each model
        Returns:
            (np.array): array of shape = [n_samples]
        Raises:
            AttributeError: if not self.using_proba
        '''
        if not self.using_proba:
            raise AttributeError(f"The aggregation_function object proba_argmax does not support using_proba=False")

        proba_average = sum(proba)/len(self.list_models)
        def get_class(x):
            return self.list_classes[x]
        get_class_v = np.vectorize(get_class)
        return get_class_v(np.argmax(proba_average,axis=1))

    def majority_vote(self, predictions) -> pd.Series:
        '''A majority voting system of multiple predictions is used.
        In the case of a tie, we use the first model's prediction (even if it is not in the first votes)

        Args:
            (pd.Series) : the Series containing the predictions of the various models
            (pd.Series) : series in which the values are the lists of underlying model predictions
        '''
        if self.multi_label == True:
            self.logger.warning("majority_vote n'est pas compatible avec le multi-label")
        votes = predictions.value_counts().sort_values(ascending=False)
        if len(votes)>1:
            if votes.iloc[0]==votes.iloc[1]:
                return predictions[0]
            else:
                return votes.index[0]
        else:
            return votes.index[0]

    def save(self, json_data: dict = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        Raises:
            TypeError: if the json_data object is not of type dict
        '''
        if json_data is None:
            json_data = {}

        #One gives the agregated model responsible for the save
        json_data['agregated_model'] = os.path.split(self.model_dir)[-1]

        #Save each model
        list_models = []
        for model in self.list_real_models:
            model.save(json_data = json_data.copy())
            list_models.append(os.path.split(model.model_dir)[-1])
        self.list_models = list_models.copy()

        json_data['list_models'] = list_models.copy()

        # Save
        list_real_models = self.list_real_models
        self.list_real_models = None
        super().save(json_data=json_data)
        self.list_real_models = list_real_models

    def get_and_save_metrics(self, y_true, y_pred, x=None, series_to_add: List[pd.Series] = None, type_data: str = '', model_logger=None) -> None:
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
            pd.DataFrame:  the df which contains the statistics
        '''
        if series_to_add is not None:
            if sum([1 if type(_) == pd.Series else 0 for _ in series_to_add]) != len(series_to_add):
                raise TypeError("L'objet series_to_add doit être composé de pd.Series uniquement")

        for model in self.list_real_models:
            model.get_and_save_metrics(y_true=y_true,
                                     y_pred=y_pred,
                                     x=x,
                                     series_to_add=series_to_add,
                                     type_data=type_data,
                                     model_logger=model_logger)


        super().get_and_save_metrics(y_true=y_true,
                                     y_pred=y_pred,
                                     x=x,
                                     series_to_add=series_to_add,
                                     type_data=type_data,
                                     model_logger=model_logger)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")