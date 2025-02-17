#!/usr/bin/env python3

## Model TFIDF SVM

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
# - ModelTfidfSvm -> Model for predictions via TF-IDF + SVM


import os
import json
import pickle
import logging
import numpy as np
from typing import Union

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from {{package_name}} import utils
from {{package_name}}.models_training.model_pipeline import ModelPipeline
from {{package_name}}.models_training.utils_super_documents import TfidfVectorizerSuperDocuments


class ModelTfidfSvm(ModelPipeline):
    '''Model for predictions via TF-IDF + SVM'''

    _default_name = 'model_tfidf_svm'

    def __init__(self, tfidf_params: Union[dict, None] = None, svc_params: Union[dict, None] = None,
                 multiclass_strategy: Union[str, None] = None, **kwargs) -> None:
        '''Initialization of the class (see ModelPipeline & ModelClass for more arguments)

        Kwargs:
            tfidf_params (dict) : Parameters for the tfidf
            svc_params (dict) : Parameters for the SVC
            multiclass_strategy (str): Multi-classes strategy, 'ovr' (OneVsRest), or 'ovo' (OneVsOne). If None, use the default of the algorithm.
        Raises:
            ValueError: If multiclass_strategy is not 'ovo', 'ovr' or None
            ValueError: If with_super_documents and multi_label
        '''
        if multiclass_strategy is not None and multiclass_strategy not in ['ovo', 'ovr']:
            raise ValueError(f"The value of 'multiclass_strategy' must be 'ovo' or 'ovr' (not {multiclass_strategy})")
        # Init.
        super().__init__(**kwargs)

        if self.with_super_documents and self.multi_label:
            raise ValueError("The method with super documents does not support multi-labels")

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Manage model
        if tfidf_params is None:
            tfidf_params = {}
        self.tfidf = TfidfVectorizer(**tfidf_params) if not self.with_super_documents else TfidfVectorizerSuperDocuments(**tfidf_params)
        if svc_params is None:
            svc_params = {}
        self.svc = LinearSVC(**svc_params)
        self.multiclass_strategy = multiclass_strategy

        # Can't do multi-labels / multi-classes
        if not self.multi_label:
            # If not multi-classes : no impact
            if multiclass_strategy == 'ovr':
                self.pipeline = Pipeline([('tfidf', self.tfidf), ('svc', OneVsRestClassifier(self.svc))])
            elif multiclass_strategy == 'ovo':
                self.pipeline = Pipeline([('tfidf', self.tfidf), ('svc', OneVsOneClassifier(self.svc))])
            else:
                self.pipeline = Pipeline([('tfidf', self.tfidf), ('svc', self.svc)])

        # Manage multi-labels -> add a MultiOutputClassifier
        # The SVC does not natively support multi-labels
        if self.multi_label:
            self.pipeline = Pipeline([('tfidf', self.tfidf), ('svc', MultiOutputClassifier(self.svc))])

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.ndarray:
        '''Predicts the probabilities on the test set
        - /!\\ THE SVM DOES NOT RETURN PROBABILITIES, HERE WE SIMULATE PROBABILITIES EQUAL TO 0 OR 1 /!\\ -

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        if not self.multi_label:
            preds = self.pipeline.predict(x_test)
            # Format ['a', 'b', 'c', 'a', ..., 'b']
            # Transform to "proba"
            transform_dict = {col: [0. if _ != i else 1. for _ in range(len(self.list_classes))] for i, col in enumerate(self.list_classes)}
            probas = np.array([transform_dict[x] for x in preds])
        else:
            preds = self.pipeline.predict(x_test)
            # Already right format, but in int !
            probas = np.array([[float(_) for _ in x] for x in preds])
        return probas

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def decision_function(self, x_test, **kwargs) -> np.ndarray:
        '''Predict confidence scores for samples

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
        Returns:
            (?): Array, shape = [n_samples]
        '''
        if self.multi_label:
            raise ValueError("The method 'decision_function' is not compatible with a multi-labels case.")
        return self.pipeline.decision_function(x_test)

    def get_predict_position(self, x_test, y_true, **kwargs) -> np.ndarray:
        '''Gets the order of predictions of y_true.
        Positions start at 1 (not 0)

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
            y_true (?): Array-like, shape = [n_samples, n_features]
        Returns:
            np.ndarray: Array, shape = [n_samples]
        '''
        self.logger.warning("Warning, the method get_predict_position is not suitable for a SVM model"
                            "(no probabilities, we use 1 or 0)")
        return super().get_predict_position(x_test, y_true)

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save model
        if json_data is None:
            json_data = {}

        # No need to save the parameters of the pipeline steps, it is already done in ModelPipeline
        json_data['multiclass_strategy'] = self.multiclass_strategy

        # Save
        super().save(json_data=json_data)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            sklearn_pipeline_path (str): Path to standalone pipeline
        Raises:
            ValueError: If configuration_path is None
            ValueError: If sklearn_pipeline_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object sklearn_pipeline_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        sklearn_pipeline_path = kwargs.get('sklearn_pipeline_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if sklearn_pipeline_path is None:
            raise ValueError("The argument sklearn_pipeline_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(sklearn_pipeline_path):
            raise FileNotFoundError(f"The file {sklearn_pipeline_path} does not exist")

        # Load confs
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
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
                          'multiclass_strategy', 'with_super_documents']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Reload pipeline
        with open(sklearn_pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)

        # Reload pipeline elements
        self.tfidf = self.pipeline['tfidf']

        # Manage multi-labels or multi-classes
        if not self.multi_label and self.multiclass_strategy is None:
            self.svc = self.pipeline['svc']
        else:
            self.svc = self.pipeline['svc'].estimator


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
