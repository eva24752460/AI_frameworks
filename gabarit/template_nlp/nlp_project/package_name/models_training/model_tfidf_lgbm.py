#!/usr/bin/env python3

## Model TFIDF LGBM
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
# - ModelTfidfLgbm -> Model for predictions via TF-IDF + LGBM


import os
import json
import pickle
import logging
import numpy as np
from typing import Union

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from {{package_name}} import utils
from {{package_name}}.models_training.model_pipeline import ModelPipeline
from {{package_name}}.models_training.utils_super_documents import TfidfVectorizerSuperDocuments


class ModelTfidfLgbm(ModelPipeline):
    '''Model for predictions via TF-IDF + LGBM'''

    _default_name = 'model_tfidf_lgbm'

    def __init__(self, tfidf_params: Union[dict, None] = None, lgbm_params: Union[dict, None] = None,
                 multiclass_strategy: Union[str, None] = None, **kwargs) -> None:
        '''Initialization of the class (see ModelPipeline & ModelClass for more arguments)

        Kwargs:
            tfidf_params (dict) : Parameters for the tfidf
            lgbm_params (dict) : Parameters for the lgbm
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
        if lgbm_params is None:
            lgbm_params = {}
        self.lgbm = LGBMClassifier(**lgbm_params)
        self.multiclass_strategy = multiclass_strategy

        # Can't do multi-labels / multi-classes
        if not self.multi_label:
            # If not multi-classes : no impact
            if multiclass_strategy == 'ovr':
                self.pipeline = Pipeline([('tfidf', self.tfidf), ('lgbm', OneVsRestClassifier(self.lgbm))])
            elif multiclass_strategy == 'ovo':
                self.pipeline = Pipeline([('tfidf', self.tfidf), ('lgbm', OneVsOneClassifier(self.lgbm))])
            else:
                self.pipeline = Pipeline([('tfidf', self.tfidf), ('lgbm', self.lgbm)])

        # Manage multi-labels -> add a MultiOutputClassifier
        # The LGBM does not natively support multi-labels
        if self.multi_label:
            self.pipeline = Pipeline([('tfidf', self.tfidf), ('lgbm', MultiOutputClassifier(self.lgbm))])

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.ndarray:
        '''Probabilities prediction on the test dataset
            'ovo' can't predict probabilities. By default we return 1 if it is the predicted class, 0 otherwise.

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        # Use super() of Pipeline class if != 'ovo' or multi-labels
        if self.multi_label or self.multiclass_strategy != 'ovo':
            return super().predict_proba(x_test, **kwargs)
        # We return 1 if predicted, otherwise 0
        else:
            preds = self.pipeline.predict(x_test)
            # Format ['a', 'b', 'c', 'a', ..., 'b']
            # Transform to "proba"
            transform_dict = {col: [0. if _ != i else 1. for _ in range(len(self.list_classes))] for i, col in enumerate(self.list_classes)}
            probas = np.array([transform_dict[x] for x in preds])
        return probas

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
            configs = json.load(f)  # Can't set int as keys in json, so need to cast it after reloading
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
            self.lgbm = self.pipeline['lgbm']
        else:
            self.lgbm = self.pipeline['lgbm'].estimator


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
