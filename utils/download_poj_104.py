import gdown
import os

if __name__ == "__main__":
    url = "https://drive.google.com/uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU&export=download"
    output = "data/poj_104.tgz"
    gdown.download(url, output, quiet=False)
