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
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, aggregation_function='vote_labels')
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
        list_models = [ModelTfidfSvm(multi_label=True), ModelTfidfGbt()]
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

    def test03_model_aggregation_check_trained(self):
        '''Test of the method _sort_model_type of {{package_name}}.models_training.model_aggregation.ModelAggregation._check_trained'''

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
        dict_classes_int = {0: 0, 1: 1, 2: 2}
        dict_classes_str = {0: 'non', 1: 'oui'}

        # int
        svm, gbt, _, _ = self.create_svm_gbt()
        model = ModelAggregation(model_dir=model_dir)
        self.assertFalse(model.trained)
        self.assertTrue(model.list_classes is None)
        self.assertTrue(model.dict_classes is None)
        svm.fit(x_train, y_train_int)
        gbt.fit(x_train, y_train_int)
        model._sort_model_type([svm, gbt])
        model._check_trained()
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
        model._sort_model_type([svm, gbt])
        model._check_trained()
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
        gbt.fit(x_train, y_train_int)
        model._sort_model_type([svm, gbt])
        model._check_trained()
        self.assertTrue(model.trained)
        self.assertTrue(len(model.list_classes), n_classes_int)
        self.assertEqual(model.list_classes, list_classes_int)
        self.assertEqual(model.dict_classes, dict_classes_int)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # Error
        svm, gbt, _, _ = self.create_svm_gbt()
        svm.fit(['int1', 'int2'], [1, 0])
        gbt.fit(['str1', 'str2'], ['non', 'oui'])
        model = ModelAggregation(model_dir=model_dir)
        model._sort_model_type([svm, gbt])
        with self.assertRaises(TypeError):
            model._check_trained()
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
        y_train_mono = np.array(['oui', 'non', 'oui', 'non', 'none'])
        y_train_multi = pd.DataFrame({'test1': [0, 1, 0, 1, 0], 'test2': [1, 0, 0, 0, 1], 'test3': [1, 0, 1, 1, 0]})
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
            self.assertTrue(submodel.trained)
            self.assertEqual(submodel.nb_fit, 1)
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
            self.assertTrue(submodel.trained)
            self.assertEqual(submodel.nb_fit, 1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        ############################################
        # multi_label
        ############################################

        # All sub-models are multi_label
        svm, gbt, _, _ = self.create_svm_gbt(svm_param={'multi_label': True}, gbt_param={'multi_label': True})
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='all_predictions')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi[cols])
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for submodel in model.list_real_models:
            self.assertTrue(submodel.trained)
            self.assertEqual(submodel.nb_fit, 1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # Some sub-models are mono_label
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
            self.assertTrue(submodel.trained)
            self.assertEqual(submodel.nb_fit, 1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        svm, gbt, _, _ = self.create_svm_gbt(gbt_param={'multi_label': True})
        gbt.fit(x_train, y_train_multi[cols])
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='all_predictions')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for submodel in model.list_real_models:
            self.assertTrue(submodel.trained)
            self.assertEqual(submodel.nb_fit, 1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # All sub-models are mono_label
        svm, gbt, _, _ = self.create_svm_gbt()
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='all_predictions')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for submodel in model.list_real_models:
            self.assertTrue(submodel.trained)
            self.assertEqual(submodel.nb_fit, 1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        ############################################
        # Error
        ############################################

        # if model needs multi_label but y_train is y_train_mono
        list_models = [ModelTfidfSvm(multi_label=True), ModelTfidfGbt(multi_label=True)]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='all_predictions')
        with self.assertRaises(ValueError):
            model.fit(x_train, y_train_mono)
        remove_dir(model_dir)

        # if model needs mono_label but y_train is multi_label
        list_models = [ModelTfidfSvm(), ModelTfidfGbt()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=False)
        with self.assertRaises(ValueError):
            model.fit(x_train, y_train_multi[cols])
        remove_dir(model_dir)

    def test05_model_aggregation_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.model_aggregation.ModelAggregation'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        #################################################
        # Set Vas mono_label
        #################################################

        dic_mono = {'x_train': np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"]),
                    'x_test': np.array(["ici test", "là, rien!"]),
                    'y_train_1': [0, 1, 0, 1, 2],
                    'y_train_2': [1, 0, 0, 2, 3],
                    'target_1': [1, 2],
                    'target_2': [2, 3],
                    'target_probas_1': [[0, 1, 0], [0, 0, 1]],
                    'target_probas_1_2_2': [[0, 1/3, 2/3, 0], [0, 0, 1/3, 2/3]],
                    'target_multi_1': [[0, 1, 0], [0, 0, 1]],
                    'target_multi_1_2_2_all': [[0, 1, 1, 0], [0, 0, 1, 1]],
                    'target_multi_1_2_2_vote': [[0, 0, 1, 0], [0, 0, 0, 1]]}

        list_model_mono = [ModelTfidfSvm(), ModelTfidfSvm(), ModelTfidfSvm(), ModelTfidfSvm()]
        for i in range(2):
            list_model_mono[i].fit(dic_mono['x_train'], dic_mono['y_train_1'])
            list_model_mono[i+2].fit(dic_mono['x_train'], dic_mono['y_train_2'])

        def test_predict_probas(model, x_test, shape_preds, shape_probas, target_predict, target_probas):
            preds = model.predict(x_test)
            probas = model.predict(x_test, return_proba=True)
            self.assertEqual(preds.shape, shape_preds)
            self.assertEqual(probas.shape, shape_probas)
            if not model.multi_label:
                self.assertAlmostEqual(probas.sum(), len(x_test))
            self.assertTrue((preds == target_predict).all())
            self.assertTrue((probas == target_probas).all())

        #################################################
        # aggregation_function: majority_vote
        # not usint_proba
        # not multi_label
        #################################################

        # All models have the same labels
        list_models = [list_model_mono[0], list_model_mono[1]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='majority_vote')
        test_predict_probas(model, dic_mono['x_test'],
                            shape_preds=(len(dic_mono['x_test']),),
                            shape_probas=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']))),
                            target_predict=dic_mono['target_1'], target_probas=dic_mono['target_probas_1'])
        remove_dir(model_dir)

        # The models have different labels
        list_models = [list_model_mono[0], list_model_mono[2], list_model_mono[3]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='majority_vote')
        test_predict_probas(model, dic_mono['x_test'],
                            shape_preds=(len(dic_mono['x_test']),),
                            shape_probas=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']+dic_mono['y_train_2']))),
                            target_predict=dic_mono['target_2'], target_probas=dic_mono['target_probas_1_2_2'])
        remove_dir(model_dir)

        #################################################
        # aggregation_function: proba_argmax
        # usint_proba
        # not multi_label
        #################################################

        # All models have the same labels
        list_models = [list_model_mono[0], list_model_mono[1]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='proba_argmax')
        test_predict_probas(model, dic_mono['x_test'],
                            shape_preds=(len(dic_mono['x_test']),),
                            shape_probas=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']))),
                            target_predict=dic_mono['target_1'], target_probas=dic_mono['target_probas_1'])
        remove_dir(model_dir)

        # The models have different labels
        list_models = [list_model_mono[0], list_model_mono[2], list_model_mono[3]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='proba_argmax')
        test_predict_probas(model, dic_mono['x_test'],
                            shape_preds=(len(dic_mono['x_test']),),
                            shape_probas=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']+dic_mono['y_train_2']))),
                            target_predict=dic_mono['target_2'], target_probas=dic_mono['target_probas_1_2_2'])
        remove_dir(model_dir)

        #################################################
        # aggregation_function: Callable
        # not usint_proba
        # not multi_label
        #################################################

        def function_test(predictions):
            labels, counts = np.unique(predictions, return_counts=True)
            votes = [(label, count) for label, count in zip(labels, counts)]
            votes = sorted(votes, key=lambda x: x[1], reverse=True)
            if len(votes) > 1 and votes[0][1] == votes[1][1]:
                return predictions[0]
            else:
                return votes[0][0]

        # All models have the same labels
        list_models = [list_model_mono[0], list_model_mono[1]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, multi_label=False, aggregation_function=function_test)
        test_predict_probas(model, dic_mono['x_test'],
                            shape_preds=(len(dic_mono['x_test']),),
                            shape_probas=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']))),
                            target_predict=dic_mono['target_1'], target_probas=dic_mono['target_probas_1'])
        remove_dir(model_dir)

        # The models have different labels
        list_models = [list_model_mono[0], list_model_mono[2], list_model_mono[3]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, multi_label=False, aggregation_function=function_test)
        test_predict_probas(model, dic_mono['x_test'],
                            shape_preds=(len(dic_mono['x_test']),),
                            shape_probas=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']+dic_mono['y_train_2']))),
                            target_predict=dic_mono['target_2'], target_probas=dic_mono['target_probas_1_2_2'])
        remove_dir(model_dir)

        #################################################
        # Set Vas multi_label
        #################################################

        dic_multi = {'x_train': np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"]),
                    'x_test': np.array(["ici test", "là, rien!"]),
                    'y_train_1': pd.DataFrame({'test1': [0, 1, 0, 1, 0], 'test2': [1, 0, 0, 1, 0], 'test3': [0, 0, 0, 1, 1]}),
                    'y_train_2': pd.DataFrame({'oui': [1, 1, 0, 1, 0], 'non': [1, 0, 1, 0, 0]}),
                    'y_train_mono': ['test3', 'test3', 'test3', 'test3', 'test4'],
                    'cols_1': ['test1', 'test2', 'test3'],
                    'cols_2': ['oui', 'non'],
                    'target_1': [[1, 1, 1], [0, 0, 1]],
                    'target_1_mono_all': [[1, 1, 1, 0], [0, 0, 1, 1]],
                    'target_1_mono_vote': [[1, 1, 1, 0], [0, 0, 1, 0]],
                    'target_2_all': [[0, 1, 1, 1, 1], [0, 0, 0, 0, 1]],
                    'target_2_vote': [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
                    'target_probas_1': [[1, 1, 1], [0, 0, 1]],
                    'target_probas_1_1_mono': [[2/3, 2/3, 1, 0], [0, 0, 2/3, 1/3]],
                    'target_probas_1_2_2': [[0, 2/3, 1/3, 1/3, 1/3], [0, 0, 0, 0, 1/3]]}

        list_model_multi = [ModelTfidfSvm(multi_label=True), ModelTfidfSvm(multi_label=True), ModelTfidfSvm(multi_label=True), ModelTfidfSvm(multi_label=True)]
        for i in range(2):
            list_model_multi[i].fit(dic_multi['x_train'], dic_multi['y_train_1'][dic_multi['cols_1']])
            list_model_multi[i+2].fit(dic_multi['x_train'], dic_multi['y_train_2'][dic_multi['cols_2']])
        svm_mono = ModelTfidfSvm()
        svm_mono.fit(dic_multi['x_train'], dic_multi['y_train_mono'])

        #################################################
        # aggregation_function: all_predictions
        # not usint_proba
        # mono and multi_label
        #################################################

        ##### All models have the same labels (mono_label)
        list_models = [list_model_mono[0], list_model_mono[1]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='all_predictions')
        test_predict_probas(model, dic_multi['x_test'],
                            shape_preds=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']))),
                            shape_probas=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']))),
                            target_predict=dic_mono['target_multi_1'], target_probas=dic_mono['target_probas_1'])
        remove_dir(model_dir)

        ##### The models have different labels (mono_label)
        list_models = [list_model_mono[0], list_model_mono[2], list_model_mono[3]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='all_predictions')
        test_predict_probas(model, dic_mono['x_test'],
                            shape_preds=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']+dic_mono['y_train_2']))),
                            shape_probas=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']+dic_mono['y_train_2']))),
                            target_predict=dic_mono['target_multi_1_2_2_all'], target_probas=dic_mono['target_probas_1_2_2'])
        remove_dir(model_dir)

        ##### All models have the same labels (multi_label)
        list_models = [list_model_multi[0], list_model_multi[1]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='all_predictions')
        test_predict_probas(model, dic_multi['x_test'],
                            shape_preds=(len(dic_multi['x_test']), len(set(dic_multi['cols_1']))),
                            shape_probas=(len(dic_multi['x_test']), len(set(dic_multi['cols_1']))),
                            target_predict=dic_multi['target_1'], target_probas=dic_multi['target_probas_1'])
        remove_dir(model_dir)

        ##### The models have different labels (multi_label)
        list_models = [list_model_multi[0], list_model_multi[2], list_model_multi[3]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='all_predictions')
        test_predict_probas(model, dic_multi['x_test'],
                            shape_preds=(len(dic_multi['x_test']), len(set(dic_multi['cols_1'] + dic_multi['cols_2']))),
                            shape_probas=(len(dic_multi['x_test']), len(set(dic_multi['cols_1'] + dic_multi['cols_2']))),
                            target_predict=dic_multi['target_2_all'], target_probas=dic_multi['target_probas_1_2_2'])
        remove_dir(model_dir)

        ##### The models have different labels (mono_label and multi_label)
        list_models = [list_model_multi[0], list_model_multi[1], svm_mono]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='all_predictions')
        test_predict_probas(model, dic_multi['x_test'],
                            shape_preds=(len(dic_multi['x_test']), len(set(dic_multi['cols_1'] + dic_multi['y_train_mono']))),
                            shape_probas=(len(dic_multi['x_test']), len(set(dic_multi['cols_1'] + dic_multi['y_train_mono']))),
                            target_predict=dic_multi['target_1_mono_all'], target_probas=dic_multi['target_probas_1_1_mono'])
        remove_dir(model_dir)

        #################################################
        # aggregation_function: vote_labels
        # not usint_proba
        # mono and multi_label
        #################################################

        ##### All models have the same labels (mono_label)
        list_models = [list_model_mono[0], list_model_mono[1]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='vote_labels')
        test_predict_probas(model, dic_multi['x_test'],
                            shape_preds=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']))),
                            shape_probas=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']))),
                            target_predict=dic_mono['target_multi_1'], target_probas=dic_mono['target_probas_1'])
        remove_dir(model_dir)

        ##### The models have different labels (mono_label)
        list_models = [list_model_mono[0], list_model_mono[2], list_model_mono[3]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='vote_labels')
        test_predict_probas(model, dic_mono['x_test'],
                            shape_preds=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']+dic_mono['y_train_2']))),
                            shape_probas=(len(dic_mono['x_test']), len(set(dic_mono['y_train_1']+dic_mono['y_train_2']))),
                            target_predict=dic_mono['target_multi_1_2_2_vote'], target_probas=dic_mono['target_probas_1_2_2'])
        remove_dir(model_dir)

        ##### All models have the same labels (multi_label)
        list_models = [list_model_multi[0], list_model_multi[1]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='vote_labels')
        test_predict_probas(model, dic_multi['x_test'],
                            shape_preds=(len(dic_multi['x_test']), len(set(dic_multi['cols_1']))),
                            shape_probas=(len(dic_multi['x_test']), len(set(dic_multi['cols_1']))),
                            target_predict=dic_multi['target_1'], target_probas=dic_multi['target_probas_1'])
        remove_dir(model_dir)

        ##### The models have different labels (multi_label)
        list_models = [list_model_multi[0], list_model_multi[2], list_model_multi[3]]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='vote_labels')
        test_predict_probas(model, dic_multi['x_test'],
                            shape_preds=(len(dic_multi['x_test']), len(set(dic_multi['cols_1'] + dic_multi['cols_2']))),
                            shape_probas=(len(dic_multi['x_test']), len(set(dic_multi['cols_1'] + dic_multi['cols_2']))),
                            target_predict=dic_multi['target_2_vote'], target_probas=dic_multi['target_probas_1_2_2'])
        remove_dir(model_dir)

        ##### The models have different labels (mono_label and multi_label)
        list_models = [list_model_multi[0], list_model_multi[1], svm_mono]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='vote_labels')
        test_predict_probas(model, dic_multi['x_test'],
                            shape_preds=(len(dic_multi['x_test']), len(set(dic_multi['cols_1'] + dic_multi['y_train_mono']))),
                            shape_probas=(len(dic_multi['x_test']), len(set(dic_multi['cols_1'] + dic_multi['y_train_mono']))),
                            target_predict=dic_multi['target_1_mono_vote'], target_probas=dic_multi['target_probas_1_1_mono'])
        remove_dir(model_dir)

        for i in range(len(list_model_mono)):
            remove_dir(list_model_mono[i].model_dir)
        for i in range(len(list_model_multi)):
            remove_dir(list_model_multi[i].model_dir)
        remove_dir(svm_mono.model_dir)

        ############################################
        # Error
        ############################################

        # Model needs to be fitted
        list_models = [ModelTfidfSvm(), ModelTfidfGbt()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        with self.assertRaises(AttributeError):
            model.predict('test')
        remove_dir(model_dir)

    def test06_model_aggregation_get_proba(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation._get_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        n_classes = 3

        list_models = [ModelTfidfSvm(), ModelTfidfGbt()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        model.fit(x_train, y_train_mono)
        probas = model._get_probas(x_train)
        self.assertTrue(isinstance(probas, np.ndarray))
        self.assertEqual(len(probas), len(x_train))
        self.assertEqual(probas.shape, (len(x_train), len(list_models), n_classes))
        svm = ModelTfidfSvm()
        gbt = ModelTfidfGbt()
        svm.fit(x_train, y_train_mono)
        gbt.fit(x_train, y_train_mono)
        probas_svm = svm.predict_proba(x_train)
        probas_gbt = gbt.predict_proba(x_train)
        self.assertTrue(([probas[i][0] for i in range(len(probas))] == probas_svm).all())
        self.assertTrue(([probas[i][1] for i in range(len(probas))] == probas_gbt).all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(svm.model_dir)
        remove_dir(gbt.model_dir)

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
        list_models = [ModelTfidfSvm(), ModelTfidfGbt()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        model.fit(x_train, y_train_mono)
        preds = model._get_predictions(x_train)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(len(preds), len(x_train))
        svm = ModelTfidfSvm()
        gbt = ModelTfidfGbt()
        svm.fit(x_train, y_train_mono)
        gbt.fit(x_train, y_train_mono)
        preds_svm = svm.predict(x_train)
        preds_gbt = gbt.predict(x_train)
        self.assertTrue(([preds[i][0] for i in range(len(x_train))] == preds_svm).all())
        self.assertTrue(([preds[i][1] for i in range(len(x_train))] == preds_gbt).all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(svm.model_dir)
        remove_dir(gbt.model_dir)

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

        list_models = [ModelTfidfSvm(), ModelTfidfGbt()]
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
        self.assertTrue((svm1_predict_full_classes == target_predict_model_with_full_list_classes).all())
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
        self.assertTrue((svm1_predict_full_classes == target_predict_model_with_full_list_classes).all())
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
        self.assertTrue((all_predictions_0 == target_all_predictions[0]).all())
        self.assertTrue((np.array([model.all_predictions(array) for array in preds]) == target_all_predictions).all())
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
        self.assertTrue((vote_labels_0 == target_vote_labels_1[0]).all())
        self.assertTrue((np.array([model.vote_labels(array) for array in preds]) == target_vote_labels_1).all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

    def test14_model_aggregation_save(self):
        '''Test of the method save of {{package_name}}.models_training.model_aggregation.ModelAggregation.save'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

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

        #######################
        #  mono_label
        #######################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])

        # Create model
        list_models = [ModelTfidfSvm(), ModelTfidfGbt()]
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
        self.assertTrue((model.predict(x_test) == model_new.predict(x_test)).all())
        self.assertTrue((model.predict_proba(x_test) == model_new.predict_proba(x_test)).all())
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
        self.assertTrue((model.predict(x_test) == model_new.predict(x_test)).all())
        self.assertTrue((model.predict_proba(x_test) == model_new.predict_proba(x_test)).all())
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