from utils.params import ParamsManager
from utils.memory import Memory

PARAMS_FILE = "params.json"


class BreakOutPlayer:
    def __init__(self):
        paramsManager = ParamsManager(PARAMS_FILE)
        memory = Memory(total_stack_depth=paramsManager.get_params()["total_stack_depth"],
                        batch_size=paramsManager.get_params()["batch_size"],
                        transition_stack_depth=paramsManager.get_params()["transition_stack_depth"])



if __name__ == '__main__':
    BreakoutPlayer = BreakOutPlayer()