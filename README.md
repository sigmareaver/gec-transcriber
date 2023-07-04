- Convert txt data (in the form of `Name: Message`) to csv
- `-s`: Strip HTML tags
- `-u`: Convert unicode punctuation to ASCII nearest equivalent, discard the rest  (may add unidecode later, and optionally discard unicode characters that can't be converted)
- `-f`: Use fastpunct to help restore any missing punctuation.
- Run a GEC model over the dataset (prompt dependent)

Convert model to CTranslate2:<br>
```
python main.py ../full-models/coedit-large convert models/coedit-large -q int8_float16
```

Process dataset:<br>
```
python main.py coedit-large transcribe ../datasets/my_dataset -d cuda -su -b 32
```
