class RecordService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    state = dict()
    current_layer = -1

    def reset(self):
        self.state = dict()
        self.current_layer = -1

    def get(self, key):
        return self.state[key]

    def set(self, key, value):
        self.state[key] = value

    def set_current_layer(self, number):
        self.current_layer = number
