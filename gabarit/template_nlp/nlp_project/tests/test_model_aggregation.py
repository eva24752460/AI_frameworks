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
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.list_models, list_models)
        self.assertTrue(os.path.isdir(model_dir))
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        # aggregation_function is 'proba_argmax'
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
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
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models={}, multi_label=True)
        remove_dir(model_dir)


    def test02_model_aggregation_get_real_models(self):
        '''Test of the method _get_real_models of tfidfDemo.models_training.model_aggregation.ModelAggregation._get_real_models'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model_name = 'test_model_name'
        
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, model_name=model_name)
        self.assertEqual(model.list_real_models, list_models)
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

        # # Multi-labels
        # list_models = [ModelTfidfSvm(multi_label=True), ModelTfidfGbt(multi_label=True)]
        # model = ModelAggregation(model_dir=model_dir, list_models=list_models)
        # self.assertFalse(model.trained)
        # self.assertEqual(model.nb_fit, 0)
        # model.fit(x_train, y_train_multi[cols])
        # self.assertTrue(model.trained)
        # self.assertEqual(model.nb_fit, 1)
        # remove_dir(model_dir)

        # Error
        with self.assertRaises(RuntimeError):
            svm = ModelTfidfSvm()
            svm.fit(x_train, y_train_mono)
            list_models = [svm, ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models)
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

        # Function majority_vote
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_mono)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train),))
        preds = model.predict('test')
        self.assertEqual(preds, model.predict(['test'])[0])
        remove_dir(model_dir)

        # Function proba_argmax
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
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
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models)
            model.predict_proba('test')
        remove_dir(model_dir)

    def test05_model_aggregation_get_proba(self):
        '''Test of tfidfDemo.models_training.model_aggregation.ModelAggregation._get_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])

        # model using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_mono)
        probas = model._get_probas(x_train)
        self.assertTrue(type(probas) == list)
        self.assertEqual(len(probas), 2)

        model_svm = ModelTfidfSvm()
        model_svm.fit(x_train, y_train_mono)
        probas_svm = model_svm.predict_proba(x_train)
        self.assertEqual(probas[0].all(), probas_svm.all())
        remove_dir(model_dir)
        remove_dir(model_svm.model_dir)

        # model not using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_mono)
        probas = model._get_probas(x_train)
        self.assertTrue(type(probas) == list)
        self.assertEqual(len(probas), 2)

        model_svm = ModelTfidfSvm()
        model_svm.fit(x_train, y_train_mono)
        probas_svm = model_svm.predict_proba(x_train)
        self.assertEqual(probas[0].all(), probas_svm.all())
        remove_dir(model_dir)
        remove_dir(model_svm.model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
            model._get_probas('test')
        remove_dir(model_dir)


    def test06_model_aggregation_get_predictions(self):
        '''Test of tfidfDemo.models_training.model_aggregation.ModelAggregation._get_predictions'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])

        # model not using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_mono)
        preds = model._get_predictions(x_train)
        self.assertTrue(type(preds) == dict)
        self.assertEqual(len(preds), 2)

        model_svm = ModelTfidfSvm()
        model_svm.fit(x_train, y_train_mono)
        preds_svm = model_svm.predict(x_train)
        self.assertEqual(preds[0].all(), preds_svm.all())
        remove_dir(model_dir)
        remove_dir(model_svm.model_dir)

        # model using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_mono)
        preds = model._get_predictions(x_train)
        self.assertTrue(type(preds) == dict)
        self.assertEqual(len(preds), 2)
        model_svm = ModelTfidfSvm()
        model_svm.fit(x_train, y_train_mono)
        preds_svm = model_svm.predict(x_train)
        self.assertEqual(preds[0].all(), preds_svm.all())
        remove_dir(model_dir)
        remove_dir(model_svm.model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models)
            model._get_predictions('test')
        remove_dir(model_dir)

    def test07_model_aggregation_predict_proba(self):
        '''Test of tfidfDemo.models_training.model_aggregation.ModelAggregation.predict_proba'''

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
        probas = model.predict_proba(x_train)
        self.assertEqual(len(probas), len(x_train))
        self.assertEqual(len(probas[0]), n_classes)
        remove_dir(model_dir)

        # model not using_proba
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_mono)
        probas = model.predict_proba(x_train)
        self.assertEqual(len(probas), len(x_train))
        self.assertEqual(len(probas[0]), n_classes)
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models)
            model.predict_proba('test')
        remove_dir(model_dir)

    def test08_model_aggregation_proba_argmax(self):
        '''Test of tfidfDemo.models_training.model_aggregation.ModelAggregation.proba_argmax'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        y_train_str = np.array(['oui', 'oui', 'non', 'oui', 'non'])

        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_mono)
        get_probas = model.proba_argmax(model._get_probas(x_train))
        self.assertEqual(len(get_probas), len(x_train))
        remove_dir(model_dir)

        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_str)
        get_probas = model.proba_argmax(model._get_probas(x_train))
        self.assertEqual(len(get_probas), len(x_train))
        remove_dir(model_dir)

        # Model needs to be using_proba = True
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models)
            model.fit(x_train, y_train_mono)
            probas_list = model._get_probas(x_train)
            model_new = ModelAggregation(list_models=list_models)
            model_new.proba_argmax(probas_list)
        remove_dir(model_dir)
        remove_dir(model_new.model_dir)

    def test09_model_aggregation_majority_vote(self):
        '''Test of tfidfDemo.models_training.model_aggregation.ModelAggregation.majority_vote'''

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
        remove_dir(model_dir)

        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_str)
        get_proba = pd.DataFrame(model._get_predictions(x_train))
        self.assertEqual(model.majority_vote(pd.DataFrame(get_proba)[0]), 'oui' or 'non')
        remove_dir(model_dir)

        # Model needs to be using_proba = False
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
            model.fit(x_train, y_train_mono)
            get_proba = pd.DataFrame(model._get_predictions(x_train))
            self.assertEqual(type(model.majority_vote(pd.DataFrame(get_proba)[0])), type(y_train_mono[0]))
        remove_dir(model_dir)


    def test10_model_aggregation_save(self):
        '''Test of the method save of tfidfDemo.models_training.model_aggregation.ModelAggregation.save'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # With proba_argmax
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.save(json_data={'test': 10})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 10)
        self.assertTrue('mainteners' in configs.keys())
        self.assertTrue('date' in configs.keys())#
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
        remove_dir(model_dir)

        # With majority_vote
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.save(json_data={'test': 10})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
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
        remove_dir(model_dir)

        # Model needs to be using_proba = False
        with self.assertRaises(ValueError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models)
            model.save(json_data='test')
        remove_dir(model_dir)

    def test11_model_aggregation_get_and_save_metrics(self):
        '''Test of the method tfidfDemo.models_training.model_aggregation.ModelAggregation.get_and_save_metrics'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Set vars
        x_train = np.array(["pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1])

        # get_and_save_metrics - mono-label
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        model.fit(x_train, y_train_mono)
        model.list_classes = [0, 1,]
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
        remove_dir(model_dir)

        # # get_and_save_metrics - multi-labels
        # list_models = [ModelTfidfSvm(multi_label=True), ModelTfidfGbt(multi_label=True)]
        # model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=True, aggregation_function='proba_argmax')
        # model.list_classes = ['test1', 'test2', 'test3']
        # y_true = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
        # y_pred = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]])
        # df_metrics = model.get_and_save_metrics(y_true, y_pred)
        # self.assertEqual(df_metrics.shape[0], 3) # 2 classes + All
        # self.assertEqual(df_metrics.loc[2, :]['Label'], 'All')
        # self.assertEqual(df_metrics.loc[2, :]['Accuracy'], 0.5)
        # plots_path = os.path.join(model.model_dir, 'plots')
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'predictions.csv')))
        # self.assertTrue(os.path.exists(os.path.join(plots_path, 'confusion_matrix_normalized.png')))
        # self.assertTrue(os.path.exists(os.path.join(plots_path, 'confusion_matrix.png')))

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
            model = ModelAggregation(model_dir=model_dir, list_models=list_models)
            df_metrics = model.get_and_save_metrics(y_true, y_pred)
        remove_dir(model_dir)

    def test12_model_aggregation_reload_from_standalone(self):
        '''Test of the method tfidfDemo.models_training.model_aggregation.ModelAaggregation.reload_from_standalone'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        ############################################
        # mono_label & with function majority_vote
        ############################################

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])

        # Create model
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model = ModelAggregation(model_dir=model_dir, list_models=list_models, using_proba=False, aggregation_function='majority_vote')
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        model_aggregation_path = os.path.join(model.model_dir, "model_aggregation.pkl")
        list_models = [ModelTfidfSvm(), ModelTfidfSuperDocumentsNaive()]
        model_new = ModelAggregation()
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, model_aggregation_path=model_aggregation_path)
        print('  ....> test model_new list_models :', model_new.list_real_models)

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
        self.assertEqual(model.list_real_models, model_new.list_real_models)
        self.assertEqual(model.list_models_names, model_new.list_models_names)
        self.assertEqual(model.using_proba, model_new.using_proba)

        self.assertEqual(model.aggregation_function, model_new.aggregation_function)
        remove_dir(model_dir)
        remove_dir(model_new.model_dir)

        ############################################
        # mono_label & with function proba_argmax
        ############################################

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
        model_aggregation_path = os.path.join(model.model_dir, "model_aggregation.pkl")
        model_new = ModelAggregation()
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, model_aggregation_path=model_aggregation_path)

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
        self.assertEqual(model.list_real_models, model_new.list_real_models)
        self.assertEqual(model.list_models_names, model_new.list_models_names)
        self.assertEqual(model.using_proba, model_new.using_proba)
        self.assertEqual(model.aggregation_function, model_new.aggregation_function)
        remove_dir(model_dir)
        remove_dir(model_new.model_dir)


        ############################################
        # Errors
        ############################################

        with self.assertRaises(FileNotFoundError):
            new_model = ModelAggregation()
            new_model.reload_from_standalone(model_dir=model_dir, configuration_path='toto.json', model_aggregation_path=model_aggregation_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelAggregation()
            new_model.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, model_aggregation_path='toto.pkl')

# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()