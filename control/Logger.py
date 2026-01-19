
class Logger:
    def __init__(self, name: str):
        self.name = name

    def log(self, *args):
        print(f"[{self.name}] ", args)
