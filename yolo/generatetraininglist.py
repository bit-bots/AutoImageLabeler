import glob
import os

imagefiletypes = ["jpg","png"]
files = []
absolutepath = os.path.abspath(".") 
for filetype in imagefiletypes:
    files.extend(glob.glob("*/*." + filetype))

files = [os.path.join(absolutepath, fi) for fi in files]

with open("train.txt", "w+") as f:
    for e in files:
        f.write(e + "\n")
