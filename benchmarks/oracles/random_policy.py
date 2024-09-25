
class RandomPolicy():

    def __init__(self):
        self.is_success = False
        self.action_types = []

    def act(self, info=None, environment=None):
        action_space = info['action_space']
        return action_space.sample()

    def get_action_type(self):
        return 'random'
    
    def get_name(self):
        return 'random'
    
    def init(self, info):
        pass

    def update(self, info, action):
        pass

    def success(self):
        return False
    
    def get_phase(self):
        return 'random'
    
    def terminate(self):
        return False
    
    def load(self):
        return -1
    
    def get_state(self):
        return {}