from pathlib  import Path
import json

if __name__ == '__main__':
    pathlist = Path('tweets').glob('*.json')
    for path in pathlist:
        print(path)
        _dir = path.with_suffix('')
        _dir.mkdir(parents=True, exist_ok=True)
        with path.open() as tweetsFile:
            tweets = json.load(tweetsFile)
            transit = _dir / "transit"
            transit.mkdir(parents=True, exist_ok=True)
            not_transit = _dir / "not_transit"
            not_transit.mkdir(parents=True, exist_ok=True)
            for i in range(len(tweets)):
                tweetPath = _dir / (str(i)+'.txt')
                with tweetPath.open('w', encoding="utf-8") as tweetFile:
                    tweetFile.write(tweets[i])
