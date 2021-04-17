from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

if __name__ == "__main__":

    zipurl = 'http://go.vicos.si/kolektorsdd2'

    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall('.')
