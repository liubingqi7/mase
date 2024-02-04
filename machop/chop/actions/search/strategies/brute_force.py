import torch
import pandas as pd
import logging
from tabulate import tabulate
import joblib

from functools import partial
from .base import SearchStrategyBase

from chop.passes.module.analysis import calculate_avg_bits_module_analysis_pass

logger = logging.getLogger(__name__)


class SearchStrategyBruteForce(SearchStrategyBase):
    is_iterative = False

    def _post_init_setup(self):
        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        if not self.sum_scaled_metrics:
            self.directions = [
                self.config["metrics"][k]["direction"] for k in self.metric_names
            ]
        else:
            self.direction = self.config["setup"]["direction"]

    def search(self, search_space):
        # construct the search list
        search_list = []      
        for choice in search_space.choices_flattened.items():
            if len(choice[1]) > 1:
                search_list.append(list(range(len(choice[1]))))
        print
        for name, length in search_space.choice_lengths_flattened.items():
            sampled_indexes[name] = trial.suggest_int(name, 0, length - 1)


        # return 

