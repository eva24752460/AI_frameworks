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
from unittest.mock import Mock
from unittest.mock import patch

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

from sklearn.svm import SVC


# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelTfidfaggregation(unittest.TestCase):
    '''Main class to test model_aggregation'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_aggregation_init(self):
        '''Test of tfidfDemo.models_training.model_aggregation.ModelAggregation.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all parameters
        list_models = [SVC(), ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.list_models, list_models)
        self.assertTrue(os.path.isdir(model_dir))
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        # aggregation_function is 'proba_argmax'
        list_models = [SVC(), ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        self.assertTrue(type(model.list_models) == list)
        self.assertTrue(type(model.using_proba) == bool)
        self.assertTrue(model.using_proba)
        self.assertTrue(model.list_real_models == list_models)
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        self.assertFalse(model.using_proba)
        remove_dir(model_dir)

        # Error
        with self.assertRaises(TypeError):
            model = ModelAggregation(model_dir=model_dir)
        remove_dir(model_dir)
        with self.assertRaises(TypeError):
            model = ModelAggregation(model_dir=model_dir, list_models={}, aggregation_function=1234)
        remove_dir(model_dir)
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models={}, aggregation_function='1234')
        remove_dir(model_dir)
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models={}, aggregation_function='majority_vote', using_proba=True)
        remove_dir(model_dir)
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models={}, aggregation_function='proba_argmax', using_proba=False)
        remove_dir(model_dir)

    def test02_model_aggregation__get_real_models(self):
        '''Test of the method _get_real_models of tfidfDemo.models_training.model_aggregation.ModelAggregation'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model_name = 'test_model_name'
        
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, model_name=model_name)
        self.assertEqual(model.list_real_models, list_models)

        # maybe for load model
        # model.save()
        # list_models_dir = ['model_test_123456789']
        # model_new = ModelAggregation(list_models=list_models_dir)
        # self.assertEqual(model.list_real_models, model_new.list_real_models)
        remove_dir(model_dir)
    
    def test03_model_aggregation_fit(self):
        '''Test of the method fit of tfidfDemo.models_training.model_aggregation.ModelAggregation'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 0])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']

        # Mono-label
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        remove_dir(model_dir)

        # Multi-labels
        list_models = [ModelTfidfSvm(multi_label=True), ModelTfidfGbt(multi_label=True)]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi[cols])
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        remove_dir(model_dir)

        # Error
        with self.assertRaises(RuntimeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models)
            model.fit(x_train, y_train_mono)
            model.fit(x_train, y_train_mono)
        remove_dir(model_dir)

    def test04_model_aggregation_predict(self):
        '''Test of the method predict of tfidfDemo.models_training.model_aggregation.ModelAggregation'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']

        # Mono-label - no strategy
        list_models = [ModelTfidfSvm(multiclass_strategy=None), ModelTfidfSuperDocumentsNaive(multiclass_strategy=None)]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        model.fit(x_train, y_train_mono)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train),))
        preds = model.predict('test')
        self.assertEqual(preds, model.predict(['test'])[0])
        remove_dir(model_dir)

        # Mono-label - using_proba
        list_models = [ModelTfidfSvm(multiclass_strategy=None), ModelTfidfSuperDocumentsNaive(multiclass_strategy=None)]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_mono)
        proba = model.predict(x_train)
        self.assertEqual(proba.shape, (len(x_train),))
        proba = model.predict('test')
        self.assertEqual([proba], model.predict(['test']))
        remove_dir(model_dir)

        # # Multi-labels
        # list_models = [ModelTfidfSvm(multi_label=True), ModelTfidfGbt(multi_label=True)]
        # model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        # model.fit(x_train, y_train_multi[cols])
        # preds = model.predict(x_train)
        # self.assertEqual(preds.shape, (len(x_train), len(cols)))
        # preds = model.predict('test')
        # self.assertEqual([elem for elem in preds], [elem for elem in model.predict(['test'])[0]])
        # remove_dir(model_dir)

        # Multi-label - using_proba
        # list_models = [ModelTfidfSvm(multi_label=True), ModelTfidfGbt(multi_label=True)]
        # model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        # model.fit(x_train, y_train_multi[cols])
        # proba = model.predict(x_train)
        # self.assertEqual(proba.shape, (len(x_train),))
        # proba = model.predict('test')
        # self.assertEqual([elem for elem in preds], [elem for elem in model.predict(['test'])[0]])
        # remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelAggregation(model_dir=model_dir, multi_label=False, multiclass_strategy=None)
            model.predict_proba('test')
        remove_dir(model_dir)

    def test05_model_aggregation_get_proba(self):
        '''Test of tfidfDemo.models_training.model_aggregation.ModelAggregation._get_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])

        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        model.fit(x_train, y_train_mono)
        probas = model._get_probas(x_train)
        self.assertTrue(type(probas) == list)
        self.assertEqual(len(probas), 2)

        model_svm = ModelTfidfSvm()
        model_svm.fit(x_train, y_train_mono)
        probas_svm = model_svm.predict_proba(x_train)
        self.assertEqual(probas[0].all(), probas_svm.all())
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models)
            model._get_probas('test')
        remove_dir(model_dir)

# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()