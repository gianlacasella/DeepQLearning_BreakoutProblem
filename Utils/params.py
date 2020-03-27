import json


class ParamsManager(object):
    def __init__(self, params_file):
        try:
            self.params = json.load(open(params_file, 'r'))
        except FileNotFoundError:
            print("[!] ERROR: ", params_file, " not found")


    def get_params(self):
        return self.params
