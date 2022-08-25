#!/usr/bin/env python3
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

# Libs unittest
import unittest

# Utils libs
import os
import json
import shutil
import numpy as np
import pandas as pd

from {{package_name}} import utils
from {{package_name}}.models_training.model_tfidf_svm import ModelTfidfSvm
from {{package_name}}.models_training.model_tfidf_gbt import ModelTfidfGbt
from {{package_name}}.models_training.model_aggregation import ModelAggregation
from {{package_name}}.models_training.model_tfidf_super_documents_naive import ModelTfidfSuperDocumentsNaive

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class Modelaggregation(unittest.TestCase):
    '''Main class to test model_aggregation'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    # Create and save a ModelTfidfSvm and a ModelTfidfGbt models
    def create_svm_gbt(self, svm_param: dict = None, gbt_param: dict = None):
        model_path = utils.get_models_path()
        model_dir_svm = os.path.join(model_path, 'model_test_123456789_svm')
        model_dir_gbt = os.path.join(model_path, 'model_test_123456789_gbt')
        remove_dir(model_dir_svm)
        remove_dir(model_dir_gbt)

        svm_param = {} if svm_param is None else svm_param
        gbt_param = {} if gbt_param is None else gbt_param
        svm = ModelTfidfSvm(model_dir=model_dir_svm, **svm_param)
        gbt = ModelTfidfGbt(model_dir=model_dir_gbt, **gbt_param)

        svm.save()
        gbt.save()
        svm_name = os.path.split(svm.model_dir)[-1]
        gbt_name = os.path.split(gbt.model_dir)[-1]
        return svm, gbt, svm_name, gbt_name

    # Create and fit models
    def mock_model_mono_multi_int_str_fitted(self, dict_var: dict = None):
        if dict is None:
            # Set vars
            x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
            y_train_int = np.array([0, 1, 0, 1, 2])
            y_train_str = np.array(['oui', 'non', 'oui', 'non', 'oui'])
            y_train_multi_1 = pd.DataFrame({'test1': [0, 1, 0, 1, 0], 'test2': [1, 0, 0, 1, 0], 'test3': [0, 0, 0, 1, 1]})
            y_train_multi_2 = pd.DataFrame({'oui': [0, 1, 0, 1, 0], 'non': [1, 0, 1, 0, 0]})
            cols_1 = ['test1', 'test2', 'test3']
            cols_2 = ['oui', 'non']
            # Set dict_var (argument for mock_model_mono_multi_int_str_fitted function)
            dict_var = {'x_train': x_train, 'y_train_int': y_train_int, 'y_train_str': y_train_str,
                        'y_train_multi_1': y_train_multi_1, 'y_train_multi_2': y_train_multi_2, 'cols_1': cols_1, 'cols_2': cols_2}

        # create models
        model_mono_int, model_mono_str, _, _ = self.create_svm_gbt()
        model_multi1, model_multi2, _, _ = self.create_svm_gbt(svm_param={'multi_label': True}, gbt_param={'multi_label': True})

        x_train = dict_var['x_train']
        y_train_multi_1 = dict_var['y_train_multi_1']
        y_train_multi_2 = dict_var['y_train_multi_2']

        model_mono_int.fit(x_train, dict_var['y_train_int'])
        model_mono_str.fit(x_train, dict_var['y_train_str'])
        model_multi1.fit(x_train, y_train_multi_1[dict_var['cols_1']])
        model_multi2.fit(x_train, y_train_multi_2[dict_var['cols_2']])

        return model_mono_int, model_mono_str, model_multi1, model_multi2

    def test01_model_aggregation_init(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model_name = 'test_model_name'

        ############################################
        # Init., test all parameters
        ############################################

        # list_models = [model, model]
        # aggregation_function: proba_argmax
        # usint_proba
        # not multi_label
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=True, multi_label=False, aggregation_function='proba_argmax')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertFalse(model.multi_label)
        self.assertTrue(model.using_proba)
        self.assertTrue(isinstance(model.list_models, list))
        self.assertEqual(model.list_models, [svm_name, gbt_name])
        self.assertTrue(isinstance(model.list_real_models[0], type(svm)))
        self.assertTrue(isinstance(model.list_real_models[1], type(gbt)))
        self.assertTrue(isinstance(model._is_gpu_activated(), bool))
        model_new = ModelAggregation()
        self.assertEqual(model_new.proba_argmax.__code__.co_code, model.aggregation_function.__code__.co_code)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        model.display_if_gpu_activated()
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(model_new.model_dir)

        # list_models = [model_name, model_name]
        # aggregation_function: majority_vote
        # not usint_proba
        # not multi_label
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt_name]
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=False, multi_label=False, aggregation_function='majority_vote')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model_name, model_name)
        self.assertFalse(model.multi_label)
        self.assertFalse(model.using_proba)
        self.assertTrue(isinstance(model.list_models, list))
        self.assertEqual(model.list_models, list_models)
        self.assertTrue(isinstance(model.list_real_models[0], type(svm)))
        self.assertTrue(isinstance(model.list_real_models[1], type(gbt)))
        self.assertTrue(isinstance(model._is_gpu_activated(), bool))
        model_new = ModelAggregation()
        self.assertEqual(model_new.majority_vote.__code__.co_code, model.aggregation_function.__code__.co_code)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        model.display_if_gpu_activated()
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(model_new.model_dir)

        # list_models = [model_name, model]
        # aggregation_function: all_predictions
        # not usint_proba
        # multi_label
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=False, multi_label=True, aggregation_function='all_predictions')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model_name, model_name)
        self.assertTrue(model.multi_label)
        self.assertFalse(model.using_proba)
        self.assertTrue(isinstance(model.list_models, list))
        self.assertEqual(model.list_models, [svm_name, gbt_name])
        self.assertTrue(isinstance(model.list_real_models[0], type(svm)))
        self.assertTrue(isinstance(model.list_real_models[1], type(gbt)))
        self.assertTrue(isinstance(model._is_gpu_activated(), bool))
        model_new = ModelAggregation()
        self.assertEqual(model_new.all_predictions.__code__.co_code, model.aggregation_function.__code__.co_code)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        model.display_if_gpu_activated()
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(model_new.model_dir)

        # list_models = [model_name, model]
        # aggregation_function: vote_labels
        # not usint_proba
        # multi_label
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=False, multi_label=True, aggregation_function='vote_labels')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model_name, model_name)
        self.assertTrue(model.multi_label)
        self.assertFalse(model.using_proba)
        self.assertTrue(isinstance(model.list_models, list))
        self.assertEqual(model.list_models, [svm_name, gbt_name])
        self.assertTrue(isinstance(model.list_real_models[0], type(svm)))
        self.assertTrue(isinstance(model.list_real_models[1], type(gbt)))
        self.assertTrue(isinstance(model._is_gpu_activated(), bool))
        model_new = ModelAggregation()
        self.assertEqual(model_new.vote_labels.__code__.co_code, model.aggregation_function.__code__.co_code)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        model.display_if_gpu_activated()
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(model_new.model_dir)

        ############################################
        # Error
        ############################################

        # if the object aggregation_function is a str but not found in the dictionary dict_aggregation_function
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, aggregation_function='1234')
        remove_dir(model_dir)

        # if the object aggregation_function is not compatible with the value using_proba
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, aggregation_function='majority_vote', using_proba=True)
        remove_dir(model_dir)
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, aggregation_function='proba_argmax', using_proba=False)
        remove_dir(model_dir)
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, aggregation_function='all_predictions', using_proba=True)
        remove_dir(model_dir)
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, aggregation_function='vote_labels', using_proba=True)
        remove_dir(model_dir)

        # if the object aggregation_function is not compatible with the value multi_label
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, aggregation_function='majority_vote', multi_label=True)
        remove_dir(model_dir)
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, aggregation_function='proba_argmax', multi_label=True)
        remove_dir(model_dir)
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, aggregation_function='all_predictions', multi_label=False)
        remove_dir(model_dir)
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, aggregation_function='vote_labels', multi_label=False)
        remove_dir(model_dir)

        # if aggregation_function object is Callable and using_proba is None
        function_test = ModelAggregation().aggregation_function
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, using_proba=None, aggregation_function=function_test)
        remove_dir(model_dir)

        # if 'multi_label' inconsistent
        list_models = [ModelTfidfSvm(multi_label=True), ModelTfidfSuperDocumentsNaive()]
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=False)
        remove_dir(model_dir)

    def test02_model_aggregation_sort_model_type(self):
        '''Test of the method _sort_model_type of {{package_name}}.models_training.model_aggregation.ModelAggregation._sort_model_type'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # list_models = [model, model]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir)
        model._sort_model_type(list_models)
        self.assertTrue(isinstance(model.list_real_models[0], type(svm)))
        self.assertTrue(isinstance(model.list_real_models[1], type(gbt)))
        self.assertEqual(len(model.list_models), len(list_models))
        self.assertEqual(model.list_models, [svm_name, gbt_name])
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # list_models = [model_name, model_name]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt_name]
        model = ModelAggregation(model_dir=model_dir)
        model._sort_model_type(list_models)
        self.assertTrue(isinstance(model.list_real_models[0], type(svm)))
        self.assertTrue(isinstance(model.list_real_models[1], type(gbt)))
        self.assertEqual(len(model.list_models), len(list_models))
        self.assertEqual(model.list_models, [svm_name, gbt_name])
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # list_models = [model_name, model]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir)
        model._sort_model_type(list_models)
        self.assertTrue(isinstance(model.list_real_models[0], type(svm)))
        self.assertTrue(isinstance(model.list_real_models[1], type(gbt)))
        self.assertEqual(len(model.list_models), len(list_models))
        self.assertEqual(model.list_models, [svm_name, gbt_name])
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

    def test03_model_aggregation_set_trained(self):
        '''Test of the method _sort_model_type of {{package_name}}.models_training.model_aggregation.ModelAggregation._set_trained'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_int = np.array([0, 1, 0, 1, 2])
        y_train_str = np.array(['oui', 'non', 'oui', 'non', 'oui'])
        n_classes_int = 3
        n_classes_str = 2
        n_classes_all = 5
        list_classes_int = [0, 1, 2]
        list_classes_str = ['non', 'oui']
        list_classes_all = [0, 1, 2, 'non', 'oui']
        dict_classes_int = {0: 0, 1: 1, 2: 2}
        dict_classes_str = {0: 'non', 1: 'oui'}
        dict_classes_all = {0: 0, 1: 1, 2: 2, 3: 'non', 4: 'oui'}

        # int
        svm, gbt, _, _ = self.create_svm_gbt()
        model = ModelAggregation(model_dir=model_dir)
        self.assertFalse(model.trained)
        self.assertTrue(model.list_classes is None)
        self.assertTrue(model.dict_classes is None)
        svm.fit(x_train, y_train_int)
        gbt.fit(x_train, y_train_int)
        model.list_real_models = None
        model._sort_model_type([svm, gbt])
        model._set_trained()
        self.assertTrue(model.trained)
        self.assertTrue(len(model.list_classes), n_classes_int)
        self.assertEqual(model.list_classes, list_classes_int)
        self.assertEqual(model.dict_classes, dict_classes_int)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # str
        svm, gbt, _, _ = self.create_svm_gbt()
        model = ModelAggregation(model_dir=model_dir)
        self.assertFalse(model.trained)
        self.assertTrue(model.list_classes is None)
        self.assertTrue(model.dict_classes is None)
        svm.fit(x_train, y_train_str)
        gbt.fit(x_train, y_train_str)
        model.list_real_models = None
        model._sort_model_type([svm, gbt])
        model._set_trained()
        self.assertTrue(model.trained)
        self.assertTrue(len(model.list_classes), n_classes_str)
        self.assertEqual(model.list_classes, list_classes_str)
        self.assertEqual(model.dict_classes, dict_classes_str)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # int and str
        svm, gbt, _, _ = self.create_svm_gbt()
        model = ModelAggregation(model_dir=model_dir)
        self.assertFalse(model.trained)
        self.assertTrue(model.list_classes is None)
        self.assertTrue(model.dict_classes is None)
        svm.fit(x_train, y_train_int)
        gbt.fit(x_train, y_train_str)
        model.list_real_models = None
        model._sort_model_type([svm, gbt])
        model._set_trained()
        self.assertTrue(model.trained)
        self.assertTrue(len(model.list_classes), n_classes_all)
        self.assertEqual(model.list_classes, list_classes_all)
        self.assertEqual(model.dict_classes, dict_classes_all)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

    def test04_model_aggregation_fit(self):
        '''Test of the method fit of {{package_name}}.models_training.model_aggregation.ModelAggregation'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        ############################################
        # mono_label
        ############################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 'oui'])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']

        # not trained
        svm, gbt, _, _ = self.create_svm_gbt()
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # trained
        svm, gbt, _, _ = self.create_svm_gbt()
        svm.fit(x_train, y_train_mono)
        gbt.fit(x_train, y_train_mono)
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # some model trained
        svm, gbt, _, _ = self.create_svm_gbt()
        svm.fit(x_train, y_train_mono)
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        ############################################
        # multi_label
        ############################################

        svm, gbt, _, _ = self.create_svm_gbt(svm_param={'multi_label': True}, gbt_param={'multi_label': True})
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='all_predictions')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi[cols])
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # some model is mono_label
        svm, gbt, _, _ = self.create_svm_gbt(gbt_param={'multi_label': True})
        svm.fit(x_train, y_train_mono)
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='all_predictions')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi[cols])
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        ############################################
        # Error
        ############################################

        # if model needs mono_label but y_train is multi_label
        list_models = [ModelTfidfSvm(multi_label=True), ModelTfidfGbt(multi_label=True)]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='all_predictions')
        with self.assertRaises(ValueError):
            model.fit(x_train, y_train_mono)
        remove_dir(model_dir)

        # if model needs mono_label but y_train is multi_label
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=False)
        with self.assertRaises(ValueError):
            model.fit(x_train, y_train_multi[cols])
        remove_dir(model_dir)

    def test05_model_aggregation_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.model_aggregation.ModelAggregation'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ici test", "là, rien!"])
        y_train_int = np.array([0, 1, 0, 1, 2])
        y_train_str = np.array(['oui', 'non', 'oui', 'non', 'oui'])
        y_train_multi_1 = pd.DataFrame({'test1': [0, 1, 0, 1, 0], 'test2': [1, 0, 0, 1, 0], 'test3': [0, 0, 0, 1, 1]})
        y_train_multi_2 = pd.DataFrame({'oui': [0, 1, 0, 1, 0], 'non': [1, 0, 1, 0, 0]})
        cols_1 = ['test1', 'test2', 'test3']
        cols_2 = ['oui', 'non']
        cols_all = ['non', 'oui', 'test1', 'test2', 'test3']
        n_classes_int = len([0, 1, 2])
        n_classes_str = len(['oui', 'non'])
        n_classes_int_str = n_classes_int + n_classes_str
        # Set dict_var (argument for mock_model_mono_multi_int_str_fitted function)
        dict_var = {'x_train': x_train, 'y_train_int': y_train_int, 'y_train_str': y_train_str,
                    'y_train_multi_1': y_train_multi_1, 'y_train_multi_2': y_train_multi_2, 'cols_1': cols_1, 'cols_2': cols_2}

        # Set target (predict with x_test)
        # mono_label: models fitted with y_train_int
        target_int = np.array([1, 2])
        # mono_label: models fitted with y_train_str
        target_str = np.array(['non','oui'])
        # mono_label: models fitted with y_train_int and y_train_str
        target_probas_int_svm = np.array([[0, 1, 0], [0, 0, 1]])
        # multi_label: models fitted with y_train_int
        target_multi_int = np.array([[0, 1, 0], [0, 0, 1]])
        # multi_label: models fitted with y_train_multi_1
        target_multi_1 = np.array([[0, 1, 0, 1, 0], [1, 0, 0, 1, 0], [0, 0, 0, 1, 1]])

        #################################################
        # aggregation_function: majority_vote
        # not usint_proba
        # not multi_label
        #################################################

        model_mono_int1, model_mono_str1, _, _ = self.mock_model_mono_multi_int_str_fitted(dict_var)
        model_mono_int2, model_mono_str2, _, _ = self.mock_model_mono_multi_int_str_fitted(dict_var)

        # All models have the same labels
        list_models_int = [model_mono_int1, model_mono_int2]
        model_int = ModelAggregation(model_dir=model_dir, list_models=list_models_int, using_proba=False, multi_label=False, aggregation_function='majority_vote')
        # not return proba
        preds = model_int.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test),))
        self.assertEquals(preds.all(), target_int.all())
        # return proba
        probas = model_int.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), n_classes_int))
        self.assertAlmostEqual(probas.sum(), len(x_test))
        self.assertAlmostEqual(probas.all(), target_probas_int_svm.all())
        remove_dir(model_dir)

        # The models have different labels
        list_models_int_str = [model_mono_int1, model_mono_str1, model_mono_str2]
        model_int_str = ModelAggregation(model_dir=model_dir, list_models=list_models_int_str, using_proba=False, multi_label=False, aggregation_function='majority_vote')
        preds = model_int_str.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test),))
        self.assertTrue((preds == target_str).all())
        # return proba
        probas = model_int_str.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), n_classes_int_str))
        self.assertAlmostEqual(probas.sum(), len(x_test))
        self.assertAlmostEqual(probas.all(), target_probas_int_svm.all())
        remove_dir(model_dir)
        remove_dir(model_mono_int1.model_dir)
        remove_dir(model_mono_int2.model_dir)
        remove_dir(model_mono_str1.model_dir)
        remove_dir(model_mono_str2.model_dir)

        #################################################
        # aggregation_function: proba_argmax
        # usint_proba
        # not multi_label
        #################################################

        model_mono_int1, model_mono_str1, _, _ = self.mock_model_mono_multi_int_str_fitted(dict_var)
        model_mono_int2, model_mono_str2, _, _ = self.mock_model_mono_multi_int_str_fitted(dict_var)

        # All models have the same labels
        list_models_int = [model_mono_int1, model_mono_int2]
        model_int = ModelAggregation(model_dir=model_dir, list_models=list_models_int, using_proba=True, multi_label=False, aggregation_function='proba_argmax')
        # not return proba
        preds = model_int.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test),))
        self.assertEquals(preds.all(), target_int.all())
        # return proba
        probas = model_int.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), n_classes_int))
        self.assertAlmostEqual(probas.sum(), len(x_test))
        self.assertAlmostEqual(probas.all(), target_probas_int_svm.all())
        remove_dir(model_dir)

        # The models have different labels
        list_models_int_str = [model_mono_int1, model_mono_str1, model_mono_str2]
        model_int_str = ModelAggregation(model_dir=model_dir, list_models=list_models_int_str, using_proba=True, multi_label=False, aggregation_function='proba_argmax')
        # not return proba
        preds = model_int_str.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test),))
        self.assertTrue((preds == target_str).all())
        # return proba
        probas = model_int_str.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), n_classes_int_str))
        self.assertAlmostEqual(probas.sum(), len(x_test))
        self.assertAlmostEqual(probas.all(), target_probas_int_svm.all())
        remove_dir(model_dir)
        remove_dir(model_mono_int1.model_dir)
        remove_dir(model_mono_int2.model_dir)
        remove_dir(model_mono_str1.model_dir)
        remove_dir(model_mono_str2.model_dir)

        #################################################
        # aggregation_function: all_predictions
        # not usint_proba
        # mono and multi_label
        #################################################

        # Set target
        # models fitted with y_train_int and y_train_str
        target_multi_int_str_all = np.array([[0, 1, 0, 1, 0], [0, 0, 1, 0, 1]])
        # models fitted with y_train_multi_1 and y_train_multi_2
        target_multi_1_2_all = np.array([[0, 1, 0, 1, 0], [1, 0, 0, 1, 0], [0, 0, 0, 1, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0]])
        # models fitted with y_train_str and y_train_multi_1
        target_multi_1_str_all = np.array([[0, 1, 0, 1, 0], [1, 0, 0, 1, 0], [0, 0, 0, 1, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])

        model_mono_int1, model_mono_str1, model_multi_cols1_1, model_multi_cols2_1 = self.mock_model_mono_multi_int_str_fitted(dict_var)
        model_mono_int2, model_mono_str2, model_multi_cols1_2, _ = self.mock_model_mono_multi_int_str_fitted(dict_var)

        ##### All models have the same labels (mono_label)
        list_models_int = [model_mono_int1, model_mono_int2]
        model_multi_int = ModelAggregation(model_dir=model_dir, list_models=list_models_int, using_proba=False, multi_label=True, aggregation_function='all_predictions')
        # not return proba
        preds = model_multi_int.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test), n_classes_int))
        self.assertEquals(preds.all(), target_multi_int.all())
        # return proba
        probas = model_multi_int.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), n_classes_int))
        self.assertAlmostEqual(probas.sum(), len(x_test))
        self.assertAlmostEqual(probas.all(), target_multi_int.all())
        remove_dir(model_dir)

        ##### All models have the same labels (multi_label)
        list_models_int = [model_multi_cols1_1, model_multi_cols1_2]
        model_multi_cols = ModelAggregation(model_dir=model_dir, list_models=list_models_int, using_proba=False, multi_label=True, aggregation_function='all_predictions')
        # not return proba
        preds = model_multi_cols.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test), len(cols_1)))
        self.assertEquals(preds.all(), target_multi_1.all())
        # return proba
        probas = model_multi_cols.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), len(cols_1)))
        self.assertAlmostEqual(probas.all(), target_multi_1.all())
        remove_dir(model_dir)

        ##### The models have different labels (mono_label)
        list_models_int_str = [model_mono_int1, model_mono_str1, model_mono_str2]
        model_int_str = ModelAggregation(model_dir=model_dir, list_models=list_models_int_str, using_proba=False, multi_label=True, aggregation_function='all_predictions')
        # not return proba
        preds = model_int_str.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test), n_classes_int_str))
        self.assertTrue((preds == target_multi_int_str_all).all())
        # return proba
        probas = model_int_str.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), n_classes_int_str))
        self.assertAlmostEqual(probas.sum(), len(x_test))
        self.assertAlmostEqual(probas.all(), target_multi_int_str_all.all())
        remove_dir(model_dir)

        ##### The models have different labels (multi_label)
        list_models_int = [model_multi_cols1_1, model_multi_cols1_2, model_multi_cols2_1]
        model_multi_cols_diff = ModelAggregation(model_dir=model_dir, list_models=list_models_int, using_proba=False, multi_label=True, aggregation_function='all_predictions')
        # not return proba
        preds = model_multi_cols_diff.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test), len(cols_all)))
        self.assertEquals(preds.all(), target_multi_1_2_all.all())
        # return proba
        probas = model_multi_cols_diff.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), len(cols_all)))
        self.assertAlmostEqual(probas.all(), target_multi_1_2_all.all())
        remove_dir(model_dir)

        ##### The models have different labels (mono_label and multi_label)
        list_models_int = [model_mono_str1, model_multi_cols1_1]
        model_multi_mono = ModelAggregation(model_dir=model_dir, list_models=list_models_int, using_proba=False, multi_label=True, aggregation_function='all_predictions')
        # not return proba
        preds = model_multi_mono.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test), len(cols_1) + n_classes_str))
        self.assertEquals(preds.all(), target_multi_1_str_all.all())
        # return proba
        probas = model_multi_mono.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), len(cols_1) + n_classes_str))
        self.assertAlmostEqual(probas.all(), target_multi_1_str_all.all())
        remove_dir(model_dir)
        remove_dir(model_mono_int1.model_dir)
        remove_dir(model_mono_int2.model_dir)
        remove_dir(model_mono_str1.model_dir)
        remove_dir(model_mono_str2.model_dir)
        remove_dir(model_multi_cols1_1.model_dir)
        remove_dir(model_multi_cols2_1.model_dir)
        remove_dir(model_multi_cols1_2.model_dir)

        #################################################
        # aggregation_function: vote_labels
        # not usint_proba
        # mono and multi_label
        #################################################

        # Set target
        # models fitted with y_train_int *2 and y_train_str
        target_multi_int_str_vote = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        # models fitted with y_train_multi_1 *2 and y_train_multi_2
        target_multi_1_2_vote = np.array([[0, 1, 0, 1, 0], [1, 0, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        # models fitted with y_train_str *2 and y_train_multi_1
        target_multi_1_str_vote = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])

        model_mono_int1, model_mono_str1, model_multi_cols1_1, model_multi_cols2_1 = self.mock_model_mono_multi_int_str_fitted(dict_var)
        model_mono_int2, model_mono_str2, model_multi_cols1_2, _ = self.mock_model_mono_multi_int_str_fitted(dict_var)

        ##### All models have the same labels (mono_label)
        list_models_int = [model_mono_int1, model_mono_int2]
        model_multi_int = ModelAggregation(model_dir=model_dir, list_models=list_models_int, using_proba=False, multi_label=True, aggregation_function='vote_labels')
        # not return proba
        preds = model_multi_int.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test), n_classes_int))
        self.assertEquals(preds.all(), target_multi_int.all())
        # return proba
        probas = model_multi_int.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), n_classes_int))
        self.assertAlmostEqual(probas.sum(), len(x_test))
        self.assertAlmostEqual(probas.all(), target_multi_int.all())
        remove_dir(model_dir)

        ##### All models have the same labels (multi_label)
        list_models_int = [model_multi_cols1_1, model_multi_cols1_2]
        model_multi_cols = ModelAggregation(model_dir=model_dir, list_models=list_models_int, using_proba=False, multi_label=True, aggregation_function='vote_labels')
        # not return proba
        preds = model_multi_cols.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test), len(cols_1)))
        self.assertEquals(preds.all(), target_multi_1.all())
        # return proba
        probas = model_multi_cols.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), len(cols_1)))
        self.assertAlmostEqual(probas.all(), target_multi_1.all())
        remove_dir(model_dir)

        ##### The models have different labels (mono_label)
        list_models_int_str = [model_mono_int1, model_mono_int2, model_mono_str1]
        model_int_str = ModelAggregation(model_dir=model_dir, list_models=list_models_int_str, using_proba=False, multi_label=True, aggregation_function='vote_labels')
        # not return proba
        preds = model_int_str.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test), n_classes_int_str))
        self.assertTrue((preds == target_multi_int_str_vote).all())
        # return proba
        probas = model_int_str.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), n_classes_int_str))
        self.assertAlmostEqual(probas.sum(), len(x_test))
        self.assertAlmostEqual(probas.all(), target_multi_int_str_vote.all())
        remove_dir(model_dir)

        ##### The models have different labels (multi_label)
        list_models_int = [model_multi_cols1_1, model_multi_cols1_2, model_multi_cols2_1]
        model_multi_cols_diff = ModelAggregation(model_dir=model_dir, list_models=list_models_int, using_proba=False, multi_label=True, aggregation_function='vote_labels')
        # not return proba
        preds = model_multi_cols_diff.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test), len(cols_all)))
        self.assertEquals(preds.all(), target_multi_1_2_vote.all())
        # return proba
        probas = model_multi_cols_diff.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), len(cols_all)))
        self.assertAlmostEqual(probas.all(), target_multi_1_2_vote.all())
        remove_dir(model_dir)

        ##### The models have different labels (mono_label and multi_label)
        list_models_int = [model_mono_str1, model_mono_str2, model_multi_cols1_1]
        model_multi_mono = ModelAggregation(model_dir=model_dir, list_models=list_models_int, using_proba=False, multi_label=True, aggregation_function='vote_labels')
        # not return proba
        preds = model_multi_mono.predict(x_test)
        self.assertEqual(preds.shape, (len(x_test), len(cols_1) + n_classes_str))
        self.assertEquals(preds.all(), target_multi_1_str_vote.all())
        # return proba
        probas = model_multi_mono.predict(x_test, return_proba=True)
        self.assertEqual(probas.shape, (len(x_test), len(cols_1) + n_classes_str))
        self.assertAlmostEqual(probas.all(), target_multi_1_str_vote.all())
        remove_dir(model_dir)
        remove_dir(model_mono_int1.model_dir)
        remove_dir(model_mono_int2.model_dir)
        remove_dir(model_mono_str1.model_dir)
        remove_dir(model_mono_str2.model_dir)
        remove_dir(model_multi_cols1_1.model_dir)
        remove_dir(model_multi_cols2_1.model_dir)
        remove_dir(model_multi_cols1_2.model_dir)

        ############################################
        # Error
        ############################################

        # Model needs to be fitted
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        with self.assertRaises(AttributeError):
            model.predict_proba('test')
        remove_dir(model_dir)

    def test06_model_aggregation_get_proba(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation._get_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        n_classes = 3

        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        model.fit(x_train, y_train_mono)
        probas = model._get_probas(x_train)
        self.assertTrue(isinstance(probas, np.ndarray))
        self.assertEqual(len(probas), len(x_train))
        self.assertEqual(probas.shape, (len(x_train), len(list_models), n_classes))
        svm = ModelTfidfSvm()
        naive = ModelTfidfSuperDocumentsNaive()
        svm.fit(x_train, y_train_mono)
        naive.fit(x_train, y_train_mono)
        probas_svm = svm.predict_proba(x_train)
        probas_naive = naive.predict_proba(x_train)
        self.assertEqual(probas[0].all(), probas_svm.all())
        self.assertEqual(probas[1].all(), probas_naive.all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(svm.model_dir)
        remove_dir(naive.model_dir)

        # Model needs to be fitted
        model = ModelAggregation(model_dir=model_dir)
        with self.assertRaises(AttributeError):
            model._get_probas('test')
        remove_dir(model_dir)

    def test07_model_aggregation_get_predictions(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation._get_predictions'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array(['test1', 'test2', 'test1', 'test2', 'test0'])
        y_train_multi_1 = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        y_train_multi_2 = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test3': [1, 0, 0, 0, 0], 'test4': [0, 0, 0, 1, 0]})
        cols_1 = ['test1', 'test2', 'test3']
        cols_2 = ['test1', 'test3', 'test4']
        target_get_pred_svm = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
        target_get_pred_gbt = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]])
        n_classes_all = len(['test0', 'test1', 'test2', 'test3'])

        # mono_label
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        model.fit(x_train, y_train_mono)
        preds = model._get_predictions(x_train)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(len(preds), len(x_train))
        svm = ModelTfidfSvm()
        naive = ModelTfidfSuperDocumentsNaive()
        svm.fit(x_train, y_train_mono)
        naive.fit(x_train, y_train_mono)
        preds_svm = svm.predict(x_train)
        preds_naive = naive.predict(x_train)
        self.assertTrue(([preds[i][0] for i in range(len(x_train))] == preds_svm).all())
        self.assertTrue(([preds[i][1] for i in range(len(x_train))] == preds_naive).all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(svm.model_dir)
        remove_dir(naive.model_dir)

        # multi_label
        svm = ModelTfidfSvm(multi_label=True)
        svm.fit(x_train, y_train_multi_1[cols_1])
        list_models = [svm, ModelTfidfGbt(multi_label=True)]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='all_predictions', multi_label=True)
        model.fit(x_train, y_train_multi_2[cols_2])
        preds = model._get_predictions(x_train)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(preds.shape, (len(x_train), len(list_models), n_classes_all))
        self.assertTrue(([preds[i][0] for i in range(len(x_train))] == target_get_pred_svm).all())
        self.assertTrue(([preds[i][1] for i in range(len(x_train))] == target_get_pred_gbt).all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # Model needs to be fitted
        model = ModelAggregation(model_dir=model_dir)
        with self.assertRaises(AttributeError):
            model._get_predictions('test')
        remove_dir(model_dir)

    def test08_model_aggregation_predict_proba(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.predict_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        n_classes = 3

        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        model.fit(x_train, y_train_mono)
        probas = model.predict_proba(x_train)
        self.assertEqual(len(probas), len(x_train))
        self.assertEqual(len(probas[0]), n_classes)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # Model needs to be fitted
        model = ModelAggregation(model_dir=model_dir)
        with self.assertRaises(AttributeError):
            model.predict_proba('test')
        remove_dir(model_dir)

    def test09_model_aggregation_predict_model_with_full_list_classes(self):
        '''Test of the method save of {{package_name}}.models_training.model_aggregation.ModelAggregation._predict_model_with_full_list_classes'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        ############################################
        # mono_label
        ############################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus"])
        x_test = np.array(['ceci est un test'])
        y_mono_1 = ['test1', 'test2', 'test4']
        y_mono_2 = ['test1', 'test3', 'test4']
        cols_all = len(['test1', 'test2', 'test3', 'test4'])
        target_predict_model_with_full_list_classes = np.array([1, 0, 0, 0])
        target_predict_svm1= 'test1'

        svm1 = ModelTfidfSvm()
        svm1.fit(x_train, y_mono_1)
        svm2 = ModelTfidfSvm()
        svm2.fit(x_train, y_mono_2)

        # return_proba
        list_models = [svm1, svm2]
        model = ModelAggregation(list_models=list_models)
        svm1_predict_full_classes = model._predict_model_with_full_list_classes(svm1, x_test, return_proba=True)
        self.assertEqual(svm1_predict_full_classes.shape, (len(x_test), cols_all))
        self.assertEqual(svm1_predict_full_classes.all(), target_predict_model_with_full_list_classes.all())
        # not return_proba
        svm1_predict_full_classes = model._predict_model_with_full_list_classes(svm1, x_test, return_proba=False)
        self.assertEqual(svm1_predict_full_classes.shape, (len(x_test),))
        self.assertEqual(svm1_predict_full_classes, target_predict_svm1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        ############################################
        # multi_label
        ############################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus"])
        y_multi_1 = pd.DataFrame({'test1': [1, 0, 1], 'test2': [1, 1, 0], 'test4': [0, 1, 0]})
        cols_1 = ['test1', 'test2', 'test4']
        y_multi_2 = pd.DataFrame({'test1': [1, 1, 0], 'test3': [1, 0, 1], 'test4': [0, 1, 0]})
        cols_2 = ['test1', 'test3', 'test4']
        cols_all = len(['test1', 'test2', 'test3', 'test4'])
        target_predict_model_with_full_list_classes = np.array([1, 1, 0, 0])

        svm1 = ModelTfidfSvm(multi_label=True)
        svm1.fit(x_train, y_multi_1[cols_1])
        svm2 = ModelTfidfSvm(multi_label=True)
        svm2.fit(x_train, y_multi_2[cols_2])

        # return_proba
        model = ModelAggregation(list_models=[svm1, svm2], multi_label=True, aggregation_function='all_predictions')
        svm1_predict_full_classes = model._predict_model_with_full_list_classes(svm1, x_test, return_proba=True)
        self.assertEqual(svm1_predict_full_classes.shape, (len(x_test), cols_all))
        self.assertEqual(svm1_predict_full_classes.all(), target_predict_model_with_full_list_classes.all())
        # not return_proba
        svm1_predict_full_classes = model._predict_model_with_full_list_classes(svm1, x_test, return_proba=False)
        self.assertEqual(svm1_predict_full_classes.shape, (len(x_test), cols_all))
        self.assertEqual(svm1_predict_full_classes.all(), target_predict_model_with_full_list_classes.all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

    def test10_model_aggregation_proba_argmax(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.proba_argmax'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        target_proba_argmax_gbt = [1, 2]

        # Prepare the models
        gbt = ModelTfidfGbt()
        gbt.fit(x_train, y_train_mono)
        list_models = [gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)

        # Test
        preds = model._get_probas(x_test)
        proba_argmax_0 = model.proba_argmax(preds[0])
        self.assertEqual(proba_argmax_0, target_proba_argmax_gbt[0])
        self.assertEqual([model.proba_argmax(array) for array in preds], target_proba_argmax_gbt)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

    def test11_model_aggregation_majority_vote(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.majority_vote'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ici test", "là, rien!"])
        y_train_str_1 = np.array(['oui', 'non', 'non', 'non', 'non'])
        y_train_str_2 = np.array(['non', 'oui', 'oui', 'oui', 'oui'])
        target_majority_vote_1 = ['non', 'non']

        # Prepare the models
        svm, gbt, _, _ = self.create_svm_gbt()
        svm.fit(x_train, y_train_str_1)
        gbt.fit(x_train, y_train_str_1)
        list_models = [ModelTfidfSvm(), svm, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        model.fit(x_train, y_train_str_2)

        # Test
        preds = model._get_predictions(x_test)
        majority_vote_0 = model.majority_vote(preds[0])
        self.assertEqual(majority_vote_0, target_majority_vote_1[0])
        self.assertEqual([model.majority_vote(array) for array in preds], target_majority_vote_1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

    def test12_model_aggregation_all_predictions(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.all_predictions'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ici test", "là, rien!"])
        y_multi_1 = pd.DataFrame({'test1': [1, 0, 1, 0, 0], 'test2': [1, 1, 0, 0, 1], 'test4': [0, 1, 0, 1, 0]})
        cols_1 = ['test1', 'test2', 'test4']
        y_multi_2 = pd.DataFrame({'test1': [1, 1, 0, 0, 1], 'test3': [1, 0, 1, 1, 0], 'test4': [0, 1, 0, 0, 0]})
        cols_2 = ['test1', 'test3', 'test4']
        target_all_predictions = np.array([[0, 0, 1, 1], [1, 1, 0, 0]])

        # Prepare the models
        svm, gbt, _, _ = self.create_svm_gbt(svm_param={'multi_label': True}, gbt_param={'multi_label': True})
        svm.fit(x_train, y_multi_1[cols_1])
        gbt.fit(x_train, y_multi_2[cols_2])
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='all_predictions')

        # Test
        preds = model._get_predictions(x_test)
        all_predictions_0 = model.all_predictions(preds[0])
        self.assertEqual(all_predictions_0.all(), target_all_predictions[0].all())
        self.assertEqual(np.array([model.all_predictions(array) for array in preds]).all(), target_all_predictions.all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

    def test13_model_aggregation_vote_labels(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.vote_labels'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ici test", "là, rien!"])
        y_multi_1 = pd.DataFrame({'test1': [1, 0, 1, 0, 0], 'test2': [1, 1, 0, 0, 1], 'test4': [0, 1, 0, 1, 0]})
        cols_1 = ['test1', 'test2', 'test4']
        y_multi_2 = pd.DataFrame({'test1': [1, 1, 0, 0, 1], 'test3': [1, 0, 1, 1, 0], 'test4': [0, 1, 0, 0, 0]})
        cols_2 = ['test1', 'test3', 'test4']
        target_vote_labels_1 = np.array([[0, 0, 0, 1], [0, 1, 0, 0]])

        # Prepare the models
        svm, gbt, _, _ = self.create_svm_gbt(svm_param={'multi_label': True}, gbt_param={'multi_label': True})
        svm.fit(x_train, y_multi_1[cols_1])
        gbt.fit(x_train, y_multi_1[cols_1])
        list_models = [svm, gbt, ModelTfidfSvm(multi_label=True)]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='vote_labels')
        model.fit(x_train, y_multi_2[cols_2])

        # Test
        preds = model._get_predictions(x_test)
        vote_labels_0 = model.vote_labels(preds[0])
        self.assertEqual(vote_labels_0.all(), target_vote_labels_1[0].all())
        self.assertEqual(np.array([model.vote_labels(array) for array in preds]).all(), target_vote_labels_1.all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

    def test14_model_aggregation_save(self):
        '''Test of the method save of {{package_name}}.models_training.model_aggregation.ModelAggregation.save'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        ######################################################
        # aggregation_function = 'majority_vote'
        ######################################################

        svm, gbt, _, _ = self.create_svm_gbt()
        model = ModelAggregation(model_dir=model_dir, list_models=[svm, gbt])
        model.save(json_data={'test': 10})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 10)
        self.assertTrue('mainteners' in configs.keys())
        self.assertTrue('date' in configs.keys())
        self.assertTrue('package_version' in configs.keys())
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertTrue('model_name' in configs.keys())
        self.assertTrue('model_dir' in configs.keys())
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('multi_label' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertTrue('list_models' in configs.keys())
        self.assertTrue('using_proba' in configs.keys())
        self.assertEqual(configs['librairie'], None)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

    def test15_model_aggregation_get_and_save_metrics(self):
        '''Test of the method {{package_name}}.models_training.model_aggregation.ModelAggregation.get_and_save_metrics'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Set vars
        x_train = np.array(["pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1], 'test2': [1, 0, 0, 0], 'test3': [0, 0, 0, 1]})
        cols = ['test1', 'test2', 'test3']

        svm, gbt, _, _ = self.create_svm_gbt()
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=[svm, gbt])
        model.fit(x_train, y_train_mono)
        model.list_classes = [0, 1]
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        df_metrics = model.get_and_save_metrics(y_true, y_pred)
        self.assertEqual(df_metrics.shape[0], 3) # 2 classes + All
        self.assertEqual(df_metrics.loc[2, :]['Label'], 'All')
        self.assertEqual(df_metrics.loc[2, :]['Accuracy'], 0.5)
        plots_path = os.path.join(model.model_dir, 'plots')
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'predictions.csv')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'confusion_matrix_normalized.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'confusion_matrix.png')))
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # get_and_save_metrics - multi-labels
        svm, gbt, _, _ = self.create_svm_gbt(svm_param={'multi_label': True}, gbt_param={'multi_label': True})
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=False, aggregation_function='all_predictions', multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        model.list_classes = ['test1', 'test2', 'test3']
        y_true = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
        y_pred = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]])
        df_metrics = model.get_and_save_metrics(y_true, y_pred)
        self.assertEqual(df_metrics.shape[0], 4) # 3 classes + All
        self.assertEqual(df_metrics.loc[3, :]['Label'], 'All')
        self.assertEqual(df_metrics.loc[0, :]['Accuracy'], 1.0)
        plots_path = os.path.join(model.model_dir, 'plots')
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'predictions.csv')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'test1__confusion_matrix_normalized.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'test1__confusion_matrix.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'test2__confusion_matrix_normalized.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'test2__confusion_matrix.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'test3__confusion_matrix_normalized.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'test3__confusion_matrix.png')))
        remove_dir(model_dir)

        # Model needs to be fitted
        model = ModelAggregation(model_dir=model_dir)
        with self.assertRaises(AttributeError):
            df_metrics = model.get_and_save_metrics(y_true, y_pred)
        remove_dir(model_dir)

    def test16_model_aggregation_reload_from_standalone(self):
        '''Test of the method {{package_name}}.models_training.model_aggregation.ModelAaggregation.reload_from_standalone'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test16'

        #######################
        #  mono_label
        #######################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])

        # Create model
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        model.fit(x_train, y_train_mono)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregation()
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path)

        # Test
        self.assertEqual(model.trained, model_new.trained)
        self.assertEqual(model.nb_fit, model_new.nb_fit)
        self.assertEqual(model.x_col, model_new.x_col)
        self.assertEqual(model.y_col, model_new.y_col)
        self.assertEqual(model.list_classes, model_new.list_classes)
        self.assertEqual(model.dict_classes, model_new.dict_classes)
        self.assertEqual(model.multi_label, model_new.multi_label)
        self.assertEqual(model.level_save, model_new.level_save)
        self.assertEqual(model.list_models, model_new.list_models)
        self.assertTrue(isinstance(model.list_real_models[0], type(model_new.list_real_models[0])))
        self.assertTrue(isinstance(model.list_real_models[1], type(model_new.list_real_models[1])))
        self.assertEqual(model.using_proba, model_new.using_proba)
        self.assertEqual(model.aggregation_function.__code__.co_code, model_new.aggregation_function.__code__.co_code)
        self.assertEqual(model.predict(x_test).all(), model_new.predict(x_test).all())
        self.assertEqual(model.predict_proba(x_test).all(), model_new.predict_proba(x_test).all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        for m in model_new.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(model_new.model_dir)

        #######################
        #  multi_label
        #######################

        # Set vars
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ceci est un coucou", "pas lui", "lui non plus", "ici coucou", "là, rien!"])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        y_train_mono = np.array(['test1', 'test1', 'test3', 'test2', 'test3'])
        cols = ['test1', 'test2', 'test3']

        # Create model
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt(svm_param={'multi_label': True}, gbt_param={'multi_label': True})
        list_models = [svm_name, gbt_name]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='all_predictions')
        model.fit(x_train, y_train_multi[cols])
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregation()
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path)

        # Test
        self.assertEqual(model.trained, model_new.trained)
        self.assertEqual(model.nb_fit, model_new.nb_fit)
        self.assertEqual(model.x_col, model_new.x_col)
        self.assertEqual(model.y_col, model_new.y_col)
        self.assertEqual(model.list_classes, model_new.list_classes)
        self.assertEqual(model.dict_classes, model_new.dict_classes)
        self.assertEqual(model.multi_label, model_new.multi_label)
        self.assertEqual(model.level_save, model_new.level_save)
        self.assertEqual(model.list_models, model_new.list_models)
        self.assertTrue(isinstance(model.list_real_models[0], type(model_new.list_real_models[0])))
        self.assertTrue(isinstance(model.list_real_models[1], type(model_new.list_real_models[1])))
        self.assertEqual(model.using_proba, model_new.using_proba)
        self.assertEqual(model.aggregation_function.__code__.co_code, model_new.aggregation_function.__code__.co_code)
        self.assertEqual(model.predict(x_test).all(), model_new.predict(x_test).all())
        self.assertEqual(model.predict_proba(x_test).all(), model_new.predict_proba(x_test).all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        for m in model_new.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(model_new.model_dir)

        ############################################
        # Errors
        ############################################

        model_new = ModelAggregation()
        with self.assertRaises(FileNotFoundError):
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path='toto.json', aggregation_function_path=aggregation_function_path)
        model_new = ModelAggregation()
        with self.assertRaises(FileNotFoundError):
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path='toto.pkl')
        model_new = ModelAggregation()
        with self.assertRaises(ValueError):
            model_new.reload_from_standalone(model_dir=model_dir, aggregation_function_path=aggregation_function_path)
        model_new = ModelAggregation()
        with self.assertRaises(ValueError):
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()