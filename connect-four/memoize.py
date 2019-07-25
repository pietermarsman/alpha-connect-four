class Memoize(object):
    def __init__(self):
        self.calls = {}

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            arguments = (args, tuple(kwargs.items()))
            if arguments not in self.calls:
                self.calls[arguments] = f(*args, **kwargs)
            return self.calls[arguments]

        return wrapper
