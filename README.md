Convert model to CTranslate2:<br>
```
python main.py ../full-models/coedit-large convert models/coedit-large -q int8_float16
```

Process dataset:<br>
```
python main.py coedit-large transcribe ../datasets/my_dataset -d cuda -su -b 32
```
