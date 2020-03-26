import json


class ParamsManager(object):
    def __init__(self, params_file):
        self.params = json.load(open(params_file, 'r'))

    def get_params(self):
        return self.params
