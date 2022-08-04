#!/usr/bin/env python3

## Modèle Agrégation

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
# - ModelAgregation -> Agrégation de modèle ModelClass

import logging
import os

import numpy as np
import pandas as pd
from typing import List, Callable
from {{package_name}} import utils
from {{package_name}}.models_training import utils_models
from {{package_name}}.monitoring.model_logger import ModelLogger
from {{package_name}}.models_training.model_class import ModelClass

class ModelAgregation(ModelClass):
    '''Modèle pour agrégation de plusieurs ModelClass'''
    _default_name = 'model_agregation'

    def __init__(self, list_models: List, agregation_function='majority_vote',using_proba:bool=False, **kwargs):
        '''Initialisation de la classe (voir  ModelClass pour arguments supplémentaires)

        Args:
            list_models (list) : liste des modèles à agréger
            agregation_function (Callable or str) : fonction d'agrégation utilisée
            using_proba (bool) : dis sur quel objet on agrège (les probas ou les prédictions).
                                inutile si agregation_function est une str

        Raises:
            TypeError: si l'objet agregation_function n'est pas du type str ou Callable
            ValueError : si l'objet agregation_function qui est une str n'est pas présent dans dict_agregation_function
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        self.using_proba = using_proba

        #Get the agregation function
        dict_agregation_function = {'majority_vote':{'function':self.majority_vote,'using_proba':False},
                                    'proba_argmax':{'function':self.proba_argmax,'using_proba':True}}
        if not isinstance(agregation_function,(Callable,str)):
            raise TypeError('L\'objet agregation_function doit être du type str ou Callable.')
        if isinstance(agregation_function,str):
            if agregation_function not in dict_agregation_function.keys():
                raise ValueError(f"L'objet agregation_function ({agregation_function}) n'est pas une option valide ({dict_agregation_function.keys()})")
            self.using_proba = dict_agregation_function[agregation_function]['using_proba']
            agregation_function = dict_agregation_function[agregation_function]['function']

        self.agregation_function = agregation_function

        # Gestion modèles
        self.list_models = list_models
        self.list_real_models = None
        self._get_real_models()

    def _get_real_models(self):
        '''Populate the self.list_real_models if it is None. Also
        transforms the ModelClass in self.list_models to the
        corresponding str if need be.
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

    def fit(self, x_train, y_train, **kwargs):
        '''Entrainement du modèle
           **kwargs permet la comptabilité avec les modèles keras
        Args:
            x_train (?): array-like or sparse matrix of shape = [n_samples, n_features]
            y_train (?): array-like, shape = [n_samples, n_features]
        Raises:
            RuntimeError: si on essaie d'entrainer un modèle déjà fit
        '''
        if self.trained:
            self.logger.error("Il n'est pas prévu de pouvoir réentrainer un modèle de type agrégation")
            self.logger.error("Veuillez entrainer un nouveau modèle")
            raise RuntimeError("Impossible de réentrainer un modèle de type agrégation")

        self._get_real_models()

        # On fit chaque modèle
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
    def _get_probas(self,x_test,**kwargs):
        '''Récupère les probabilités de chacun des modèles à agréger
        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (?): array of shape = [n_samples, n_features]
        '''
        self._get_real_models()
        # On calcule les probas pour chaque modèle
        list_predict_proba = []
        for model in self.list_real_models:
            list_predict_proba.append(model.predict_proba(x_test,**kwargs))
        return list_predict_proba

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def _get_predictions(self,x_test,**kwargs):
        '''Récupère les probabilités de chacun des modèles à agréger
        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (?): dictionnaire où les valeurs sont les listes des prédictions des modèles sous-jacents
        '''
        self._get_real_models()
        dict_predict = {}
        # On prédit pour chaque modèle
        for i, model in enumerate(self.list_real_models):
            dict_predict[i] = model.predict(x_test,**kwargs)
        return dict_predict

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict(self, x_test, **kwargs):
        '''Prédictions sur test

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (?): array of shape = [n_samples]
        '''
        #On choisit si on se base sur les probas de chaque model ou si on utilise leur prédiction
        if self.using_proba:
            proba = self._get_probas(x_test,**kwargs)
            return self.agregation_function(proba)
        else:
            dict_predict = self._get_predictions(x_test,**kwargs)
            df = pd.DataFrame(dict_predict)
            # self.agregation_function est la fonction qui fait réellement le boulot d'agrégation
            df['prediction_finale'] = df.apply(lambda x:self.agregation_function(x),axis=1)
            return df['prediction_finale'].values

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs):
        '''Prédictions probabilité sur test

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (?): array of shape = [n_samples]
        '''
        list_predict_proba = self._get_probas(x_test,**kwargs)
        # On fait la moyenne des probas de tous les modèles
        return sum(list_predict_proba)/len(self.list_models)

    def proba_argmax(self, proba:List):
        '''On prend l'argmax de la moyenne des probabilités des modèles sous-jacent pour fournir une prédiction
        Args:
            proba (List) : la liste des probabilités de chacun des modèles
        '''
        proba_average = sum(proba)/len(self.list_models)
        def get_class(x):
            return self.list_classes[x]
        get_class_v = np.vectorize(get_class)
        return get_class_v(np.argmax(proba_average,axis=1))

    def majority_vote(self, predictions):
        '''Système de vote majoritaire de plusieurs prédictions.
        En cas d'égalité, on prend la prédiction du premier modèle (même s'il n'est pas dans les premiers votes)

        Args:
            predictions (pd.Series) : la Series contenant les prédictions des différents modèles
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

    def save(self, json_data: dict = None):
        '''Sauvegarde du modèle

        Kwargs:
            json_data (dict): configuration à ajouter pour la sauvegarde JSON
        Raises:
            TypeError: si l'objet json_data n'est pas du type dict
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

    def get_and_save_metrics(self, y_true, y_pred, x=None, series_to_add: List[pd.Series] = None, type_data: str = '', model_logger=None):
        '''Fonction pour obtenir et sauvegarder les métriques d'un modèle

        Args:
            y_true (?): array-like, shape = [n_samples, n_features]
            y_pred (?): array-like, shape = [n_samples, n_features]
        Kwargs:
            x (?): array-like or sparse matrix of shape = [n_samples, n_features]
            series_to_add (list): liste de pd.Series à ajouter à la dataframe
            type_data (str): type du dataset (validation, test, ...)
            model_logger (ModelLogger): classe custom pour logger les métriques dans ML Flow
        Raises:
            TypeError: si l'objet series_to_add n'est pas du type list, et composé d'éléments de type pd.Series
        Returns:
            pd.DataFrame: la df qui contient les statistiques
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
    logger.error("Ce script ne doit pas être exécuté, il s'agit d'un package.")