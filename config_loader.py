import json

class JsonReader():
    def __init__(self, path: str):
        self.path = path


    def read(self):
        # Opening JSON file
        f = open(self.path)

        # returns JSON object as
        # a dictionary
        data = json.load(f)

        # Closing file
        f.close()

        return data



