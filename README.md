# gec-transcriber
gec-transcriber is a convenience tool to assist in improving the quality of datasets, especially those that are created through automated means such as scraping.

Capabilities:
- Convert txt data (in the form of `Name: Message`) to csv
- `-c`: Specify column # to process (or `last` to specify last column, useful for groups of datasets with varying quantities of columns), otherwise process all columns
- `-s`: Strip HTML tags
- `-u`: Convert unicode punctuation to ASCII nearest equivalent
- `-U`: Convert unicode using unidecode (not implemented yet)
- `-S`: Strip unicode characters (runs after `-u` and `-U` to strip unicode characters that couldn't be converted to ASCII equivalents)
- `-f`: Use fastpunct to help restore any missing punctuation and correct spelling. Works reasonably well with low quality data, poor English, etc.
- `-p`: Specify custom prompt for GEC model
- Run a GEC model over the dataset (prompt dependent)

Parallelism:
- `-ct`: # CPU threads (default 16)
- `-gt`: # GPU threads (default 16)
- `-b`: Batch size (default 1)

Convert model to CTranslate2:<br>
```
python main.py ../full-models/coedit-large convert models/coedit-large -q int8_float16
```

Process my_dataset using GPU, where data is in column 5, and using all available processing features:<br>
```
python main.py coedit-large transcribe ../datasets/my_dataset -c 5 -d cuda -suUSf -b 32
```
Processed datasets are outputed to `/path/to/your/dataset/output`

### Installation
Clone repo and enter directory:
```
git clone https://github.com/sigmareaver/gec-transcriber.git
cd gec-transcriber
```
Create and activate an environment (e.g. with conda):
```
conda create -n gec
conda activate gec
```
Install requirements:
```
pip install -r requirements.txt
```
