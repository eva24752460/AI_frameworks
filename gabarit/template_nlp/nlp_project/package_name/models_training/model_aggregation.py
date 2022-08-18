#!/usr/bin/env python3

## Model Aggregation

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
# - ModelAggregation -> model aggregation with ModelClass

import os
import json
import logging
import dill as pickle

import numpy as np
import pandas as pd
from typing import List, Callable, Union

from {{package_name}} import utils
from {{package_name}}.models_training import utils_models
from {{package_name}}.monitoring.model_logger import ModelLogger
from {{package_name}}.models_training.model_class import ModelClass

class ModelAggregation(ModelClass):
    '''Model for aggregating multiple ModelClasses'''
    _default_name = 'model_aggregation'

    def __init__(self, list_models: List = None, aggregation_function: Union[Callable, str] = 'majority_vote', using_proba: bool = None, **kwargs) -> None:
        '''Initialization of the class (see ModelClass for more arguments)

        Args:
            list_models (list) : list of model to be aggregated
            aggregation_function (Callable or str) : aggregation function used
            using_proba (bool) : which object is being aggregated (the probas or the predictions).
        Raises:
            ValueError : if aggregation_function object is Callable and using_proba is None
            ValueError : if the object aggregation_function is a str but not found in the dictionary dict_aggregation_function
            ValueError : if the object aggregation_function is not adapte the value using_proba
            ValueError : The 'multi_label' parameters of the list models are inconsistent with the model_aggregation
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Get the aggregation function
        self.using_proba = using_proba
        dict_aggregation_function = {'majority_vote': {'function': self.majority_vote, 'using_proba': False},
                                    'proba_argmax': {'function': self.proba_argmax, 'using_proba': True}}
        if isinstance(aggregation_function, (Callable)):
            if using_proba is None:
                raise ValueError(f"When aggregation_function is Callable, using_proba(bool) cannot be None ")
        else:
            if aggregation_function not in dict_aggregation_function.keys():
                raise ValueError(f"The aggregation_function object ({aggregation_function}) is not a valid option ({dict_aggregation_function.keys()})")
            if using_proba is None:
                self.using_proba = dict_aggregation_function[aggregation_function]['using_proba']
            elif using_proba != dict_aggregation_function[aggregation_function]['using_proba']:
                raise ValueError(f"The aggregation_function object ({aggregation_function}) is not compatible with using_proba=({using_proba})")
            aggregation_function = dict_aggregation_function[aggregation_function]['function']

        # Manage model
        self.aggregation_function = aggregation_function
        self.list_real_models = None
        self.list_models = None
        if list_models is not None:
            self._sort_model_type(list_models)

        # Error for multi label inconsistency
        if self.list_real_models is not None:
            set_multi_label = {model.multi_label for model in self.list_real_models}
            if True in set_multi_label and not self.multi_label:
                    raise ValueError(f"The 'multi_label' parameters of the list models are inconsistent with the model_aggregation.")

    def _sort_model_type(self, list_models) -> None:
        '''Populate the self.list_real_models if it is None. Also transforms the ModelClass in self.list_models to the corresponding str if need be.

        Args:
            list_models (?): list of model_name or model
        '''
        if self.list_real_models is None:
            list_real_models = []
            new_list_models = []
            # Get the real model and model name
            for model in list_models:
                if isinstance(model,str):
                    real_model, _ = utils_models.load_model(model)
                    new_list_models.append(model)
                else:
                    real_model = model
                    new_list_models.append(os.path.split(model.model_dir)[-1])
                list_real_models.append(real_model)
            self.list_real_models = list_real_models
            self.list_models = new_list_models

    def fit(self, x_train, y_train, **kwargs) -> None:
        '''Trains the model
           **kwargs enables Keras model compatibility.

        Args:
            x_train (?): Array-like, shape = [n_samples]
            y_train (?): Array-like, shape = [n_samples]
        Raises:
            ValueError : if model needs mono_label but y_train is multi_label
            ValueError : if model needs multi_label but y_train is mono_label
        '''
        bool_multi_label = isinstance(y_train, pd.DataFrame) and len(y_train.iloc[0]>1)
        # Fit each model
        for model in self.list_real_models:
            if not model.trained:
                if bool_multi_label and not model.multi_label:
                    raise ValueError(f"Model ({model}) needs y_train_mono_label to fit")
                if not bool_multi_label and model.multi_label:
                    raise ValueError(f"Model ({model}) needs y_train_muliti_label to fit")
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
    def predict(self, x_test, return_proba: bool = False, **kwargs) -> np.array:
        '''Prediction

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
            return_proba (bool): If the function should return the probabilities instead of the classes
        Returns:
            (np.array): array of shape = [n_samples]
        '''
        # We decide whether to rely on each model's probas or their prediction
        if return_proba:
            return self.predict_proba(x_test)
        elif self.using_proba:
            probas = self._get_probas(x_test, **kwargs)
            preds = self.aggregation_function(probas)
            return preds
        else:
            df = self._get_predictions(x_test, **kwargs)
            df['prediction_finale'] = df.apply(self.aggregation_function, axis=1)
            return np.array(df['prediction_finale'].tolist())

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def _get_probas(self, x_test, **kwargs) -> list:
        '''Recover the probability of each model being aggregated

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (list): array of shape = [n_samples, n_features]
        '''
        # Predict for each model
        list_predict_proba = []
        for model in self.list_real_models:
            list_predict_proba.append(model.predict_proba(x_test,**kwargs))
        return list_predict_proba

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def _get_predictions(self, x_test, **kwargs) -> pd.DataFrame:
        '''Recover the probability of each model being aggregated

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (pd.DataFrame): df in which the values are lists of underlying model predictions
        '''
        list_predict = np.array([model.predict(x_test) for model in self.list_real_models])

        if not self.multi_label:
            list_predict = list_predict.T
            df = pd.DataFrame(list_predict)
        else:
            df = pd.DataFrame({key: list(vec) for key, vec in enumerate(list_predict)})
        return df

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.array:
        '''Predicts the probabilities on the test set

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (np.array): array of shape = [n_samples, n_classes]
        '''
        list_predict_proba = self._get_probas(x_test, **kwargs)
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
            raise AttributeError(f"proba_argmax is not compatible with using_proba=False")

        proba_average = sum(proba)/len(self.list_models)

        if not self.multi_label:
            def get_class(x):
                return self.list_classes[x]
            get_class_v = np.vectorize(get_class)
            return get_class_v(np.argmax(proba_average, axis=1))
        else:
            return (proba_average >= 0.5).astype(int)

    def majority_vote(self, predictions: pd.Series) -> list:
        '''Aggregation_function: A majority voting system of multiple predictions is used.
        In the case of a tie, we use the first model's prediction (even if it is not in the first votes)

        Args:
            (pd.Series) : the Series containing the predictions of each models
                          series in which the values are the lists of underlying model predictions
        Return:
            (list) : majority_vote
        Raises:
            AttributeError: if self.using_proba
        '''
        if self.using_proba:
            raise AttributeError(f"majority_vote is not compatible with using_proba=True")

        votes = predictions.value_counts().sort_values(ascending=False)
        if len(votes) > 1 and votes.iloc[0] == votes.iloc[1]:
            return predictions[0]
        else:
            return votes.index[0]

    def save(self, json_data: dict = {}) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save each model
        for model in self.list_real_models:
            model.save()

        json_data['list_models'] = self.list_models.copy()
        json_data['using_proba'] = self.using_proba

        aggregation_function = self.aggregation_function

        # Save aggregation_function if not None & level_save > LOW
        if (self.aggregation_function is not None) and (self.level_save in ['MEDIUM', 'HIGH']):
            # Manage paths
            aggregation_function_path = os.path.join(self.model_dir, "aggregation_function.pkl")
            # Save as pickle
            with open(aggregation_function_path, 'wb') as f:
                # TODO: use dill to get rid of  "can't pickle ..." errors
                pickle.dump(self.aggregation_function, f)

        # Save
        list_real_models = self.list_real_models
        delattr(self, "list_real_models")
        delattr(self, "aggregation_function")
        super().save(json_data=json_data)
        setattr(self, "aggregation_function", aggregation_function)
        setattr(self, "list_real_models", list_real_models)

    @utils.trained_needed
    def get_and_save_metrics(self, y_true, y_pred, x=None, series_to_add: List[pd.Series] = None, type_data: str = '', model_logger: ModelLogger = None) -> pd.Series:
        '''Function to obtain and save model metrics

        Args:
            y_true (?): array-like, shape = [n_samples, n_features]
            y_pred (?): array-like, shape = [n_samples, n_features]
        Kwargs:
            x (?): array-like or sparse matrix of shape = [n_samples, n_features]
            series_to_add (list): list of pd.Series to add to the dataframe
            type_data (str): dataset type (validation, test, ...)
            model_logger (ModelLogger): custom class to log metrics in ML Flow
        Returns:
            pd.DataFrame: the df which contains the statistics
        '''
        for model in self.list_real_models:
            model.get_and_save_metrics(y_true=y_true,
                                        y_pred=y_pred,
                                        x=x,
                                        series_to_add=series_to_add,
                                        type_data=type_data,
                                        model_logger=model_logger,)

        return super().get_and_save_metrics(y_true=y_true,
                                            y_pred=y_pred,
                                            x=x,
                                            series_to_add=series_to_add,
                                            type_data=type_data,
                                            model_logger=model_logger)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model aggregation from its configuration and "standalones" files
            Reloads list model from "list_models" files

        Kwargs:
            configuration_path (str): Path to configuration file
            aggregation_function_path (str): Path to aggregation_function_path
        Raises:
            ValueError: If configuration_path is None
            ValueError: If aggregation_function_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object aggregation_function_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        aggregation_function_path = kwargs.get('aggregation_function_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if aggregation_function_path is None:
            raise ValueError("The argument aggregation_function_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(aggregation_function_path):
            raise FileNotFoundError(f"The file {aggregation_function_path} does not exist")

        # Load confs
        with open(configuration_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        # Can't set int as keys in json, so need to cast it after reloading
        # dict_classes keys are always ints
        if 'dict_classes' in configs.keys():
            configs['dict_classes'] = {int(k): v for k, v in configs['dict_classes'].items()}
        elif 'list_classes' in configs.keys():
            configs['dict_classes'] = {i: col for i, col in enumerate(configs['list_classes'])}

        # Reload aggregation_function_path
        with open(aggregation_function_path, 'rb') as f:
            self.aggregation_function = pickle.load(f)

        # Set class vars
        # self.model_name = # Keep the created name
        # self.model_dir = # Keep the created folder
        self.nb_fit = configs.get('nb_fit', 1)  # Consider one unique fit by default
        self.trained = configs.get('trained', True)  # Consider trained by default
        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['x_col', 'y_col',
                          'list_classes', 'dict_classes', 'multi_label', 'level_save',
                          'list_models', 'using_proba']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        self._sort_model_type(self.list_models)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")