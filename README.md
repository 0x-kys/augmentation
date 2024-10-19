quick hacky script to generate multiple augmented images out of a single image.

```ps1
# clone this repo (http/ssh/whatever)

python -m venv venv

.\venv\Script\activate.ps1
# OR
source ./venv/bin/activate.(sh/fish) # whatever for linux

python -m pip install -r requirements.txt

# save source image as img.png
# this'll generate 100 images

python main.py
```
