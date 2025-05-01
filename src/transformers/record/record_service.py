class RecordService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    state = dict()
    current_layer_index = -1

    def reset(self):
        self.state = dict()

    def get(self, key):
        return self.state[key]

    def set(self, key, value):
        key = key.replace('LAYER_INDEX', str(self.current_layer_index))
        self.state[key] = value
