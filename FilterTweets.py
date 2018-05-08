from pathlib  import Path
import json

hasAtLeastOneInCommon = lambda a, b: any(i in b for i in a)

def deleteAll(pathlist, terms):
    for path in pathlist:
        if path.is_dir():
            deleteAll(path.iterdir(), terms)
        elif path.name != ".DS_Store":
            content = ""
            with path.open('r', encoding="utf-8") as tweetFile:
                content = tweetFile.readline()
            if hasAtLeastOneInCommon(terms, content):
                print(path)
                path.unlink()


if __name__ == '__main__':
    pathlist = Path('tweets').iterdir()
    deleteAll(pathlist, ["swarmapp"])