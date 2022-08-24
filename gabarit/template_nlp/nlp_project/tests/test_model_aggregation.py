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
    def create_svm_gbt(self, svm_param: dict = {}, gbt_param: dict = {}):
        svm = ModelTfidfSvm(**svm_param)
        gbt = ModelTfidfGbt(**gbt_param)
        svm.save()
        gbt.save()
        svm_name = os.path.split(svm.model_dir)[-1]
        gbt_name = os.path.split(gbt.model_dir)[-1]
        return svm, gbt, svm_name, gbt_name

    def test01_model_aggregation_init(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        ############################################
        # Init., test all parameters
        ############################################

        # list_models = [model, model]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertFalse(model.multi_label)
        self.assertTrue(model.using_proba)
        self.assertTrue(type(model.list_models) == list)
        self.assertEqual(model.list_models, [svm_name, gbt_name])
        self.assertEqual(type(model.list_real_models[0]), type(svm))
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        model.display_if_gpu_activated()
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # list_models = [model_name, model_name]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt_name]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertFalse(model.multi_label)
        self.assertFalse(model.using_proba)
        self.assertTrue(type(model.list_models) == list)
        self.assertEqual(model.list_models, list_models)
        self.assertEqual(type(model.list_real_models[0]), type(svm))
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        model.display_if_gpu_activated()
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # list_models = [model_name, model]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertFalse(model.multi_label)
        self.assertTrue(model.using_proba)
        self.assertTrue(type(model.list_models) == list)
        self.assertEqual(model.list_models, [svm_name, gbt_name])
        self.assertEqual(type(model.list_real_models[0]), type(svm))
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        model.display_if_gpu_activated()
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # multi_label & using_proba
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt(svm_param={'multi_label': True})
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax', multi_label=True)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertTrue(model.multi_label)
        self.assertTrue(model.using_proba)
        self.assertTrue(type(model.list_models) == list)
        self.assertEqual(model.list_models, [svm_name, gbt_name])
        self.assertEqual(type(model.list_real_models[0]), type(svm))
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        model.display_if_gpu_activated()
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # multi_label & not using_proba & function_test
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt(svm_param={'multi_label': True}, gbt_param={'multi_label': True})
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='all_predictions', multi_label=True)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertTrue(model.multi_label)
        self.assertFalse(model.using_proba)
        self.assertTrue(type(model.list_models) == list)
        self.assertEqual(model.list_models, [svm_name, gbt_name])
        self.assertEqual(type(model.list_real_models[0]), type(svm))
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        model.display_if_gpu_activated()
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        ############################################
        # Error
        ############################################

        # if the object aggregation_function is a str but not found in the dictionary dict_aggregation_function
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models={}, aggregation_function='1234')
        remove_dir(model_dir)

        # if the object aggregation_function is not adapte the value using_proba
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models={}, aggregation_function='majority_vote', using_proba=True)
        remove_dir(model_dir)
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models={}, aggregation_function='proba_argmax', using_proba=False)
        remove_dir(model_dir)

        # if aggregation_function object is Callable and using_proba is None
        with self.assertRaises(ValueError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            function_test = ModelAggregation().aggregation_function
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=None, aggregation_function=function_test)
        remove_dir(model_dir)

        # if 'multi_label' inconsistent
        with self.assertRaises(ValueError):
            list_models = [ModelTfidfSvm(multi_label=True), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=False)
        remove_dir(model_dir)

    def test02_model_aggregation_sort_model_type(self):
        '''Test of the method _sort_model_type of {{package_name}}.models_training.model_aggregation.ModelAggregation._sort_model_type'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model_name = 'test_model_name'

        # list_models = [model, model]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, model_name=model_name, aggregation_function='proba_argmax', using_proba=True)
        model._sort_model_type(list_models)
        self.assertEqual(model.list_real_models, list_models)
        self.assertEqual(model.list_models[0], svm_name)
        self.assertEqual(len(model.list_models), len(list_models))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # list_models = [model_name, model_name]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt_name]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, model_name=model_name, aggregation_function='majority_vote', using_proba=False)
        model._sort_model_type(list_models)
        self.assertEqual(type(model.list_real_models[0]), type(svm))
        self.assertEqual(model.list_models, [svm_name, gbt_name])
        self.assertEqual(len(model.list_models), len(list_models))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # list_models = [model_name, model]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='all_predictions', multi_label=True)
        self.assertEqual(type(model.list_real_models[1]), type(gbt))
        self.assertEqual(model.list_models[0], svm_name)
        self.assertEqual(len(model.list_models), len(list_models))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

    def test03_model_aggregation_fit(self):
        '''Test of the method fit of {{package_name}}.models_training.model_aggregation.ModelAggregation'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        ######################################################
        # mono_label & aggregation_funcion is Callable
        ######################################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 'oui'])

        # not using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function=lambda x: 'oui')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # using_proba and some model fitted
        svm = ModelTfidfSvm()
        svm.fit(x_train, y_train_mono)
        list_models = [svm, ModelTfidfSuperDocumentsNaive()]
        function_with_proba = ModelAggregation().proba_argmax
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function=function_with_proba)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # using_proba and all model fitted
        svm = ModelTfidfSvm()
        naive =ModelTfidfSuperDocumentsNaive()
        svm.fit(x_train, y_train_mono)
        naive.fit(x_train, y_train_mono)
        list_models = [svm, naive]
        function_with_proba = ModelAggregation().proba_argmax
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function=function_with_proba)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        ######################################################
        # mono_label & aggregation_funcion in dict_aggregation_function
        ######################################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 0])

        # aggregation_funcion is majority_vote
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # aggregation_funcion is proba_argmax
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        ######################################################
        # Multi-labels
        ######################################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 0])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']

        svm, gbt, svm_name, gbt_name = self.create_svm_gbt(svm_param={'multi_label': True})
        gbt.fit(x_train, y_train_mono)
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='all_predictions', multi_label=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi[cols])
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        remove_dir(model_dir)

        ############################################
        # Error
        ############################################

        # if model needs mono_label but y_train is multi_label
        with self.assertRaises(ValueError):
            list_models = [ModelTfidfSvm(multi_label=True), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax', multi_label=False)
            model.fit(x_train, y_train_mono)
        remove_dir(model_dir)

        # if model needs mono_label but y_train is multi_label
        with self.assertRaises(ValueError):
            list_models = [ModelTfidfSvm(multi_label=True), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax', multi_label=False)
            model.fit(x_train, y_train_multi)
        remove_dir(model_dir)

    def test04_model_aggregation_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.model_aggregation.ModelAggregation'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        ######################################################
        # mono_label & aggregation_funcion is Callable
        ######################################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_str = np.array(['oui', 'non', 'oui', 'non', 'none'])
        n_classes = 3

        # using_proba
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm, gbt]
        def function_with_proba(proba):
            proba_average = np.sum(proba, axis=0)/proba.shape[0]
            index_class = np.argmax(proba_average)
            return index_class
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function=function_with_proba)
        model.fit(x_train, y_train_str)
        preds = model.predict(x_train, return_proba=True)
        self.assertEqual(preds.shape, (len(x_train), n_classes))
        self.assertAlmostEqual(preds.sum(), len(x_train))

        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # not using_proba
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt_name]
        function_without_proba = ModelAggregation().majority_vote
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function=function_without_proba)
        model.fit(x_train, y_train_str)
        preds = model.predict(x_train, return_proba=True)
        self.assertEqual(preds.shape, (len(x_train), n_classes))
        self.assertFalse(preds[0] in y_train_str)
        self.assertAlmostEqual(preds.sum(), len(x_train))

        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        self.assertTrue(preds[0] in y_train_str)
        preds = model.predict('ceci est un test', return_proba=False)
        self.assertEqual(preds, 'oui')
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        ######################################################
        # mono_label & aggregation_funcion = 'majority_vote'
        ######################################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_str = np.array(['oui', 'non', 'oui', 'non', 'none'])
        n_classes = 3

        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm, gbt_name]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_str)
        preds = model.predict(x_train, return_proba=True)
        self.assertEqual(preds.shape, (len(x_train), n_classes))
        self.assertFalse(preds[0] in y_train_str)
        self.assertAlmostEqual(preds.sum(), len(x_train))

        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        self.assertTrue(preds[0] in y_train_str)
        preds = model.predict('ceci est un test', return_proba=False)
        self.assertEqual(preds, 'oui')
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        ######################################################
        # mono_label & aggregation_funcion = 'proba_argmax'
        ######################################################

        # list_models = [model_name, model]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_str)
        preds = model.predict(x_train, return_proba=True)
        self.assertEqual(preds.shape, (len(x_train), n_classes))
        self.assertFalse(preds[0] in y_train_str)
        self.assertAlmostEqual(preds.sum(), len(x_train))

        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        self.assertTrue(preds[0] in y_train_str)
        preds = model.predict('ceci est un test', return_proba=False)
        self.assertEqual(preds, 'oui')
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        #######################################################
        # multi_label & aggregation_funcion = 'all_predictions'
        #######################################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        n_classes = 3
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']

        svm, gbt, svm_name, gbt_name = self.create_svm_gbt(svm_param={'multi_label': True}, gbt_param={'multi_label': True})
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='all_predictions', multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), len(cols)))
        proba = model.predict('test', return_proba=True)
        self.assertEqual([elem for elem in proba], [elem for elem in model.predict(['test'], return_proba=True)[0]])
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict('test', return_proba=False)
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict(['test'], return_proba=False)[0]])
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models)
            model.predict_proba('test')
        remove_dir(model_dir)

    def test05_model_aggregation_get_proba(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation._get_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        n_classes = 3

        # model using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_mono)
        probas = model._get_probas(x_train)
        self.assertTrue(type(probas) == np.ndarray)
        self.assertEqual(len(probas), len(x_train))
        self.assertEqual(probas.shape, (len(x_train), len(list_models), n_classes))
        model_svm = ModelTfidfSvm()
        model_svm.fit(x_train, y_train_mono)
        probas_svm = model_svm.predict_proba(x_train)
        self.assertEqual(probas[0].all(), probas_svm.all())
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(model_svm.model_dir)

        # aggregation_funcion is Callable and not using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        function_without_proba = ModelAggregation().majority_vote
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function=function_without_proba)
        model.fit(x_train, y_train_mono)
        probas = model._get_probas(x_train)
        self.assertTrue(type(probas) == np.ndarray)
        self.assertEqual(len(probas), len(x_train))
        self.assertEqual(probas.shape, (len(x_train), len(list_models), n_classes))
        model_svm = ModelTfidfSvm()
        model_svm.fit(x_train, y_train_mono)
        probas_svm = model_svm.predict_proba(x_train)
        self.assertEqual(probas[0].all(), probas_svm.all())
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(model_svm.model_dir)

        # multi_label
        svm1 = ModelTfidfSvm(multi_label=True)
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus"])
        y_1 = pd.DataFrame({'test1': [1, 0, 1], 'test2': [1, 1, 0], 'test4': [0, 1, 0]})
        cols_1 = ['test1', 'test2', 'test4']
        svm1.fit(x_train, y_1[cols_1])

        svm2 = ModelTfidfSvm(multi_label=True)
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus"])
        y_2 = pd.DataFrame({'test1': [1, 1, 0], 'test3': [1, 0, 1], 'test4': [0, 1, 0]})
        cols_2 = ['test1', 'test3', 'test4']
        svm2.fit(x_train, y_2[cols_2])
        n_classes = 4 # ['test1', 'test2', 'test3', 'test4']

        list_models = [svm1, svm2]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='all_predictions')
        model.fit(x_train, y_2[cols_2])
        preds = model._get_probas(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(list_models), n_classes))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
            model._get_probas('test')
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

    def test06_model_aggregation_get_predictions(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation._get_predictions'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array(['test1', 'test2', 'test1', 'test2', 'test0'])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']
        all_multi_classes = 4 # [test0, test1, test2, test3]

        # model not using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_mono)
        preds = model._get_predictions(x_train)
        self.assertTrue(type(preds) == np.ndarray)
        self.assertEqual(len(preds), len(x_train))
        model_svm = ModelTfidfSvm()
        model_svm.fit(x_train, y_train_mono)
        preds_svm = model_svm.predict(x_train)
        self.assertTrue(([preds[i][0] for i in range(len(x_train))] == preds_svm).all())
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(model_svm.model_dir)

        # model using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_mono)
        preds = model._get_predictions(x_train)
        self.assertTrue(type(preds) == np.ndarray)
        self.assertEqual(len(preds), len(x_train))
        model_svm = ModelTfidfSvm()
        model_svm.fit(x_train, y_train_mono)
        preds_svm = model_svm.predict(x_train)
        self.assertTrue(([preds[i][0] for i in range(len(x_train))] == preds_svm).all())
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(model_svm.model_dir)

        # multi_label
        naive = ModelTfidfSuperDocumentsNaive()
        naive.fit(x_train, y_train_mono)
        list_models = [ModelTfidfSvm(multi_label=True), naive]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='all_predictions', multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        preds = model._get_predictions(x_train)
        self.assertTrue(type(preds) == np.ndarray)
        self.assertEqual(len(preds), len(x_train))
        self.assertEqual(len(preds[0][0]), all_multi_classes)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(model_svm.model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models)
            model._get_predictions('test')
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

    def test07_model_aggregation_predict_proba(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.predict_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        n_classes = 3
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']

        # aggregation_funcion is Callable
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        def argmax_sum(x):
            return np.argmax(sum(x), axis=1)
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function=argmax_sum)
        model.fit(x_train, y_train_mono)
        probas = model.predict_proba(x_train)
        self.assertEqual(len(probas), len(x_train))
        self.assertEqual(len(probas[0]), n_classes)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # model using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_mono)
        probas = model.predict_proba(x_train)
        self.assertEqual(len(probas), len(x_train))
        self.assertEqual(len(probas[0]), n_classes)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # model not using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_mono)
        probas = model.predict_proba(x_train)
        self.assertEqual(len(probas), len(x_train))
        self.assertEqual(len(probas[0]), n_classes)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # multi_label
        list_models = [ModelTfidfSvm(multi_label=True), ModelTfidfGbt(multi_label=True)]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='all_predictions', multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # multi_label with non multi_label model
        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus"])
        y_1 = pd.DataFrame({'test1': [1, 0, 1], 'test2': [1, 1, 0], 'test4': [0, 1, 0]})
        y_2 = pd.DataFrame({'test1': [1, 1, 0], 'test3': [1, 0, 1], 'test4': [0, 1, 0]})
        y_3 = np.array(['test2', 'test3', 'test4'])
        cols_1 = ['test1', 'test2', 'test4']
        cols_2 = ['test1', 'test3', 'test4']
        n_classes = 4 # ['test1', 'test2', 'test3', 'test4']

        svm1 = ModelTfidfSvm(multi_label=True)
        svm2 = ModelTfidfSvm(multi_label=True)
        gbt = ModelTfidfGbt()
        svm1.fit(x_train, y_1[cols_1])
        svm2.fit(x_train, y_2[cols_2])
        gbt.fit(x_train, y_3)

        model = ModelAggregation(model_dir=model_dir, list_models=[svm1, svm2, gbt], multi_label=True, aggregation_function='majority_vote')
        model.fit(x_train, y_2[cols_2])
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), n_classes))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models)
            model.predict_proba('test')
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

    def test08_model_aggregation_proba_argmax(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.proba_argmax'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        y_train_str = np.array(['oui', 'oui', 'non', 'oui', 'non'])

        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_mono)
        preds = model._get_probas(x_train)
        get_probas = np.array([model.proba_argmax(array) for array in preds])
        self.assertEqual(len(get_probas), len(x_train))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_str)
        preds = model._get_probas(x_train)
        get_probas = np.array([model.proba_argmax(array) for array in preds])
        self.assertEqual(len(get_probas), len(x_train))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # Model needs to be using_proba = True
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False)
            model.fit(x_train, y_train_mono)
            probas_list = model._get_probas(x_train)
            model.proba_argmax(probas_list)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

    def test09_model_aggregation_majority_vote(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.majority_vote'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        y_train_str = np.array(['oui', 'oui', 'non', 'oui', 'non'])

        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_mono)
        get_proba = pd.DataFrame(model._get_predictions(x_train))
        self.assertEqual(type(model.majority_vote(pd.DataFrame(get_proba)[0])), type(y_train_mono[0]))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_str)
        get_proba = pd.DataFrame(model._get_predictions(x_train))
        self.assertEqual(model.majority_vote(pd.DataFrame(get_proba)[0]), 'oui' or 'non')
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # Model needs to be using_proba = False
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
            model.fit(x_train, y_train_mono)
            get_proba = pd.DataFrame(model._get_predictions(x_train))
            self.assertEqual(type(model.majority_vote(pd.DataFrame(get_proba)[0])), type(y_train_mono[0]))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

    def test10_model_aggregation_all_predictions(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.all_predictions'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus"])
        y_multi_1 = pd.DataFrame({'test1': [1, 0, 1], 'test2': [1, 1, 0], 'test4': [0, 1, 0]})
        cols_1 = ['test1', 'test2', 'test4']
        y_multi_2 = pd.DataFrame({'test1': [1, 1, 0], 'test3': [1, 0, 1], 'test4': [0, 1, 0]})
        cols_2 = ['test1', 'test3', 'test4']

        svm1 = ModelTfidfSvm(multi_label=True)
        svm1.fit(x_train, y_multi_1[cols_1])
        svm2 = ModelTfidfSvm(multi_label=True)
        svm2.fit(x_train, y_multi_2[cols_2])

        result = [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]]
        y_mono = ['test1', 'test2', 'test1']

        # test
        list_models = [svm1, svm2]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='all_predictions', multi_label=True)
        model.fit(x_train, y_multi_2[cols_2])
        get_pre = model._get_predictions(x_train)
        pred = model.all_predictions(get_pre[0])
        self.assertTrue((pred == result[0]).all())
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # Model needs to be multi_label = True
        with self.assertRaises(AttributeError):
            svm1 = ModelTfidfSvm()
            svm2 = ModelTfidfSvm()
            svm1.fit(x_train, y_mono)
            svm2.fit(x_train, y_mono)
            list_models = [svm1, svm2]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='all_predictions', multi_label=False)
            model.fit(x_train, y_multi_2[cols_2])
            series = model._get_predictions(x_train)
            model.all_predictions(series.iloc[0])
            for m in model.list_real_models:
                remove_dir(os.path.split(m.model_dir)[-1])
            remove_dir(model_dir)

    def test11_model_aggregation_vote_labels(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.vote_labels'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus"])
        y_multi_1 = pd.DataFrame({'test1': [1, 0, 1], 'test2': [1, 1, 0], 'test4': [0, 1, 0]})
        cols_1 = ['test1', 'test2', 'test4']
        y_multi_2 = pd.DataFrame({'test1': [1, 1, 0], 'test3': [1, 0, 1], 'test4': [0, 1, 0]})
        cols_2 = ['test1', 'test3', 'test4']

        svm1 = ModelTfidfSvm(multi_label=True)
        svm1.fit(x_train, y_multi_1[cols_1])
        svm2 = ModelTfidfSvm(multi_label=True)
        svm2.fit(x_train, y_multi_2[cols_2])

        result = [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]]
        y_mono = ['test1', 'test2', 'test1']

        # test
        list_models = [svm1, svm2]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='vote_labels', multi_label=True)
        model.fit(x_train, y_multi_2[cols_2])
        get_pre = model._get_predictions(x_train)
        pred = model.all_predictions(get_pre[0])
        self.assertTrue((pred == result[0]).all())
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # Model needs to be multi_label = True
        with self.assertRaises(AttributeError):
            svm1 = ModelTfidfSvm()
            svm2 = ModelTfidfSvm()
            svm1.fit(x_train, y_mono)
            svm2.fit(x_train, y_mono)
            list_models = [svm1, svm2]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, aggregation_function='vote_labels', multi_label=False)
            model.fit(x_train, y_multi_2[cols_2])
            series = model._get_predictions(x_train)
            model.all_predictions(series.iloc[0])
            for m in model.list_real_models:
                remove_dir(os.path.split(m.model_dir)[-1])
            remove_dir(model_dir)


    def test12_model_aggregation_predict_model_with_full_list_classes(self):
        '''Test of the method save of {{package_name}}.models_training.model_aggregation.ModelAggregation._predict_model_with_full_list_classes'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus"])
        y_multi_1 = pd.DataFrame({'test1': [1, 0, 1], 'test2': [1, 1, 0], 'test4': [0, 1, 0]})
        cols_1 = ['test1', 'test2', 'test4']
        y_multi_2 = pd.DataFrame({'test1': [1, 1, 0], 'test3': [1, 0, 1], 'test4': [0, 1, 0]})
        cols_2 = ['test1', 'test3', 'test4']
        y_mono_3 = ['test1', 'test1', 'test5']
        cols_all = 5 # ['test1', 'test2', 'test3', 'test4', 'test5']

        svm1 = ModelTfidfSvm(multi_label=True)
        svm1.fit(x_train, y_multi_1[cols_1])
        svm2 = ModelTfidfSvm(multi_label=True)
        svm2.fit(x_train, y_multi_2[cols_2])
        svm3 = ModelTfidfSvm()
        svm3.fit(x_train, y_mono_3)

        # multi_label
        model = ModelAggregation(list_models=[svm1, svm2, svm3], multi_label=True, aggregation_function='proba_argmax')
        self.assertEqual(model._predict_model_with_full_list_classes(svm1, x_train, return_proba=False).shape, (len(x_train), cols_all))
        self.assertEqual(model._predict_model_with_full_list_classes(svm2, x_train, return_proba=True).shape, (len(x_train), cols_all))

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus"])
        y_mono_1 = ['test1', 'test2', 'test4']
        y_mono_2 = ['test1', 'test3', 'test4']
        y_mono_3 = ['test1', 'test1', 'test5']
        cols_all = 5 # ['test1', 'test2', 'test3', 'test4', 'test5']

        svm1 = ModelTfidfSvm()
        svm1.fit(x_train, y_mono_1)
        svm2 = ModelTfidfSvm()
        svm2.fit(x_train, y_mono_2)
        svm3 = ModelTfidfSvm()
        svm3.fit(x_train, y_mono_3)

        # mono_label
        model = ModelAggregation(list_models=[svm1, svm2, svm3], multi_label=False, aggregation_function='proba_argmax')
        self.assertEqual(model._predict_model_with_full_list_classes(svm2, x_train, return_proba=True).shape, (len(x_train), cols_all))

    def test13_model_aggregation_save(self):
        '''Test of the method save of {{package_name}}.models_training.model_aggregation.ModelAggregation.save'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        ######################################################
        #  aggregation_funcion is Callable
        ######################################################

        # using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function=lambda x: np.argmax(sum(x), axis=1))
        model.save(json_data={'test': 10})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))
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
        self.assertEqual(configs['librairie'], None)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # not using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        function_without_proba = ModelAggregation().majority_vote
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function=function_without_proba)
        model.save(json_data={'test': 10})

        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))
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
        self.assertEqual(configs['librairie'], None)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        ######################################################
        # aggregation_funcion = 'majority_vote'
        ######################################################

        # list_models = [model, model]
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
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
        self.assertEqual(configs['librairie'], None)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # list_models = [model_name, model_name]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt_name]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
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
        self.assertEqual(configs['librairie'], None)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # list_models = [model_name, model]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
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
        self.assertEqual(configs['librairie'], None)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        ######################################################
        # aggregation_funcion = 'proba_argmax'
        ######################################################

        # list_models = [model_name, model]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
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
        self.assertEqual(configs['librairie'], None)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        ######################################################
        # Multi label
        ######################################################

        svm, gbt, svm_name, gbt_name = self.create_svm_gbt(svm_param={'multi_label': True}, gbt_param={'multi_label': True})
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='all_predictions', multi_label=True)
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
        self.assertEqual(configs['librairie'], None)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

    def test14_model_aggregation_get_and_save_metrics(self):
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

        # get_and_save_metrics - mono-label
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
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
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

        # get_and_save_metrics - multi-labels
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt(svm_param={'multi_label': True}, gbt_param={'multi_label': True})
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=True, aggregation_function='proba_argmax', multi_label=True)
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
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models)
            df_metrics = model.get_and_save_metrics(y_true, y_pred)
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)

    def test15_model_aggregation_reload_from_standalone(self):
        '''Test of the method {{package_name}}.models_training.model_aggregation.ModelAaggregation.reload_from_standalone'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        ######################################################
        #  aggregation_funcion is Callable
        ######################################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])

        # Create model
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        def argmax_sum(x):
            return np.argmax(sum(x), axis=1)
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function=argmax_sum)
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregation()
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path)

        # Test
        self.assertEqual(model.model_name, model_new.model_name)
        self.assertEqual(model.trained, model_new.trained)
        self.assertEqual(model.nb_fit, model_new.nb_fit)
        self.assertEqual(model.x_col, model_new.x_col)
        self.assertEqual(model.y_col, model_new.y_col)
        self.assertEqual(model.list_classes, model_new.list_classes)
        self.assertEqual(model.dict_classes, model_new.dict_classes)
        self.assertEqual(model.multi_label, model_new.multi_label)
        self.assertEqual(model.level_save, model_new.level_save)
        self.assertEqual(model.list_models, model_new.list_models)
        self.assertEqual(str(model.list_real_models)[:50], str(model_new.list_real_models)[:50])
        self.assertEqual(model.using_proba, model_new.using_proba)
        exemple = [np.array([[0.8, 0.2]]), np.array([[0.1, 0.9]])]
        self.assertEqual(model.aggregation_function(exemple), model_new.aggregation_function(exemple))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        for m in model_new.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_new.model_dir)

        #######################################################################################
        # aggregation_funcion = 'majority_vote'    &    list_models = [model, model]
        #######################################################################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])

        # Create model
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_mono)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregation()
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path)

        # Test
        self.assertEqual(model.model_name, model_new.model_name)
        self.assertEqual(model.trained, model_new.trained)
        self.assertEqual(model.nb_fit, model_new.nb_fit)
        self.assertEqual(model.x_col, model_new.x_col)
        self.assertEqual(model.y_col, model_new.y_col)
        self.assertEqual(model.list_classes, model_new.list_classes)
        self.assertEqual(model.dict_classes, model_new.dict_classes)
        self.assertEqual(model.multi_label, model_new.multi_label)
        self.assertEqual(model.level_save, model_new.level_save)
        self.assertEqual(model.list_models, model_new.list_models)
        self.assertEqual(str(model.list_real_models)[:50], str(model_new.list_real_models)[:50])
        self.assertEqual(model.using_proba, model_new.using_proba)
        df = pd.DataFrame([[0, 1], [1, 1]])
        output_aggregation_fuction_model = df.apply(lambda x: model.aggregation_function(x), axis=1)
        output_aggregation_fuction_model_new = df.apply(lambda x: model_new.aggregation_function(x), axis=1)
        self.assertEqual(output_aggregation_fuction_model.all(), output_aggregation_fuction_model_new.all())
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        for m in model_new.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_new.model_dir)

        #######################################################################################
        # aggregation_funcion = 'majority_vote'    &    list_models = [model_name, model_name]
        #######################################################################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])

        # Create model
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt_name]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregation()
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path)

        # Test
        self.assertEqual(model.model_name, model_new.model_name)
        self.assertEqual(model.trained, model_new.trained)
        self.assertEqual(model.nb_fit, model_new.nb_fit)
        self.assertEqual(model.x_col, model_new.x_col)
        self.assertEqual(model.y_col, model_new.y_col)
        self.assertEqual(model.list_classes, model_new.list_classes)
        self.assertEqual(model.dict_classes, model_new.dict_classes)
        self.assertEqual(model.multi_label, model_new.multi_label)
        self.assertEqual(model.level_save, model_new.level_save)
        self.assertEqual(model.list_models, model_new.list_models)
        self.assertEqual(str(model.list_real_models)[:50], str(model_new.list_real_models)[:50])
        self.assertEqual(model.using_proba, model_new.using_proba)
        df = pd.DataFrame([[0, 1], [1, 1]])
        output_aggregation_fuction_model = df.apply(lambda x: model.aggregation_function(x), axis=1)
        output_aggregation_fuction_model_new = df.apply(lambda x: model_new.aggregation_function(x), axis=1)
        self.assertEqual(output_aggregation_fuction_model.all(), output_aggregation_fuction_model_new.all())
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        for m in model_new.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_new.model_dir)

        #######################################################################################
        # aggregation_funcion = 'majority_vote'    &    list_models = [model_name, model]
        #######################################################################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])

        # Create model
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregation()
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path)

        # Test
        self.assertEqual(model.model_name, model_new.model_name)
        self.assertEqual(model.trained, model_new.trained)
        self.assertEqual(model.nb_fit, model_new.nb_fit)
        self.assertEqual(model.x_col, model_new.x_col)
        self.assertEqual(model.y_col, model_new.y_col)
        self.assertEqual(model.list_classes, model_new.list_classes)
        self.assertEqual(model.dict_classes, model_new.dict_classes)
        self.assertEqual(model.multi_label, model_new.multi_label)
        self.assertEqual(model.level_save, model_new.level_save)
        self.assertEqual(model.list_models, model_new.list_models)
        self.assertEqual(str(model.list_real_models)[:50], str(model_new.list_real_models)[:50])
        self.assertEqual(model.using_proba, model_new.using_proba)
        df = pd.DataFrame([[0, 1], [1, 1]])
        output_aggregation_fuction_model = df.apply(lambda x: model.aggregation_function(x), axis=1)
        output_aggregation_fuction_model_new = df.apply(lambda x: model_new.aggregation_function(x), axis=1)
        self.assertEqual(output_aggregation_fuction_model.all(), output_aggregation_fuction_model_new.all())
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        for m in model_new.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_new.model_dir)

        #######################################################################################
        # aggregation_funcion = 'proba_argmax'    &    list_models = [model, model]
        #######################################################################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])

        # Create model
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregation()
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path)

        # Test
        self.assertEqual(model.model_name, model_new.model_name)
        self.assertEqual(model.trained, model_new.trained)
        self.assertEqual(model.nb_fit, model_new.nb_fit)
        self.assertEqual(model.x_col, model_new.x_col)
        self.assertEqual(model.y_col, model_new.y_col)
        self.assertEqual(model.list_classes, model_new.list_classes)
        self.assertEqual(model.dict_classes, model_new.dict_classes)
        self.assertEqual(model.multi_label, model_new.multi_label)
        self.assertEqual(model.level_save, model_new.level_save)
        self.assertEqual(model.list_models, model_new.list_models)
        self.assertEqual(str(model.list_real_models)[:50], str(model_new.list_real_models)[:50])
        self.assertEqual(model.using_proba, model_new.using_proba)
        exemple = np.array([np.array([[0.8, 0.2]]), np.array([[0.1, 0.9]])])
        self.assertEqual(model.aggregation_function(exemple), model_new.aggregation_function(exemple))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        for m in model_new.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_new.model_dir)

        #######################################################################################
        # aggregation_funcion = 'proba_argmax'    &    list_models = [model_name, model_name]
        #######################################################################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])

        # Create model
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt_name]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregation()
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path)

        # Test
        self.assertEqual(model.model_name, model_new.model_name)
        self.assertEqual(model.trained, model_new.trained)
        self.assertEqual(model.nb_fit, model_new.nb_fit)
        self.assertEqual(model.x_col, model_new.x_col)
        self.assertEqual(model.y_col, model_new.y_col)
        self.assertEqual(model.list_classes, model_new.list_classes)
        self.assertEqual(model.dict_classes, model_new.dict_classes)
        self.assertEqual(model.multi_label, model_new.multi_label)
        self.assertEqual(model.level_save, model_new.level_save)
        self.assertEqual(model.list_models, model_new.list_models)
        self.assertEqual(str(model.list_real_models)[:50], str(model_new.list_real_models)[:50])
        self.assertEqual(model.using_proba, model_new.using_proba)
        exemple = np.array([np.array([[0.8, 0.2]]), np.array([[0.1, 0.9]])])
        self.assertEqual(model.aggregation_function(exemple), model_new.aggregation_function(exemple))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        for m in model_new.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_new.model_dir)

        #######################################################################################
        # aggregation_funcion = 'proba_argmax'    &    list_models = [model_name, model]
        #######################################################################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])

        # Create model
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregation()
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path)

        # Test
        self.assertEqual(model.model_name, model_new.model_name)
        self.assertEqual(model.trained, model_new.trained)
        self.assertEqual(model.nb_fit, model_new.nb_fit)
        self.assertEqual(model.x_col, model_new.x_col)
        self.assertEqual(model.y_col, model_new.y_col)
        self.assertEqual(model.list_classes, model_new.list_classes)
        self.assertEqual(model.dict_classes, model_new.dict_classes)
        self.assertEqual(model.multi_label, model_new.multi_label)
        self.assertEqual(model.level_save, model_new.level_save)
        self.assertEqual(model.list_models, model_new.list_models)
        self.assertEqual(str(model.list_real_models)[:50], str(model_new.list_real_models)[:50])
        self.assertEqual(model.using_proba, model_new.using_proba)
        exemple = np.array([np.array([[0.8, 0.2]]), np.array([[0.1, 0.9]])])
        self.assertEqual(model.aggregation_function(exemple), model_new.aggregation_function(exemple))
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        for m in model_new.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_new.model_dir)

        #######################################################################################
        #  multi_label   &    aggregation_funcion = 'all_predictions'
        #######################################################################################

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
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='all_predictions', multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregation()
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path)

        # Test
        self.assertEqual(model.model_name, model_new.model_name)
        self.assertEqual(model.trained, model_new.trained)
        self.assertEqual(model.nb_fit, model_new.nb_fit)
        self.assertEqual(model.x_col, model_new.x_col)
        self.assertEqual(model.y_col, model_new.y_col)
        self.assertEqual(model.list_classes, model_new.list_classes)
        self.assertEqual(model.dict_classes, model_new.dict_classes)
        self.assertEqual(model.multi_label, model_new.multi_label)
        self.assertEqual(model.level_save, model_new.level_save)
        self.assertEqual(model.list_models, model_new.list_models)
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_test)], [list(_) for _ in model_new.predict_proba(x_test)])
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        for m in model_new.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_new.model_dir)

        #######################################################################################
        #  multi_label   &    aggregation_funcion is Callable
        #######################################################################################

        # Create model
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt(gbt_param={'multi_label': True})
        svm.fit(x_train, y_train_mono)
        svm.save()
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='all_predictions', multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregation()
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path)

        # Test
        self.assertEqual(model.model_name, model_new.model_name)
        self.assertEqual(model.trained, model_new.trained)
        self.assertEqual(model.nb_fit, model_new.nb_fit)
        self.assertEqual(model.x_col, model_new.x_col)
        self.assertEqual(model.y_col, model_new.y_col)
        self.assertEqual(model.list_classes, model_new.list_classes)
        self.assertEqual(model.dict_classes, model_new.dict_classes)
        self.assertEqual(model.multi_label, model_new.multi_label)
        self.assertEqual(model.level_save, model_new.level_save)
        self.assertEqual(model.list_models, model_new.list_models)
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_test)], [list(_) for _ in model_new.predict_proba(x_test)])
        for m in model.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_dir)
        for m in model_new.list_real_models:
            remove_dir(os.path.split(m.model_dir)[-1])
        remove_dir(model_new.model_dir)

        ############################################
        # Errors
        ############################################

        with self.assertRaises(FileNotFoundError):
            model_new = ModelAggregation()
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path='toto.json', aggregation_function_path=aggregation_function_path)
        with self.assertRaises(FileNotFoundError):
            model_new = ModelAggregation()
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path='toto.pkl')
        with self.assertRaises(ValueError):
            model_new = ModelAggregation()
            model_new.reload_from_standalone(model_dir=model_dir, aggregation_function_path=aggregation_function_path)
        with self.assertRaises(ValueError):
            model_new = ModelAggregation()
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()