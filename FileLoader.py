from pathlib  import Path
import json

def openElseLoad(filename, loadFunction):
    jsonPath = Path(filename)
    if jsonPath.exists():
        with jsonPath.open() as jsonFile:
            return json.load(jsonFile)
    else:
        with jsonPath.open('w') as jsonFile:
            data = loadFunction()
            json.dump(data, jsonFile)
        return data