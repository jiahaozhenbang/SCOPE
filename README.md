# SCOPE
Source code for the paper "Improving Chinese Spelling Check by Character Pronunciation Prediction: The Effects of Adaptivity and Granularity" in EMNLP 2022 

Coming soon!

## Environment
- Python: 3.8
- Cuda: 11.7
- Packages: `pip install -r requirements.txt`

## Data

### Raw Data
- SIGHAN Bake-off 2013: http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html  
- SIGHAN Bake-off 2014: http://ir.itc.ntnu.edu.tw/lre/clp14csc.html  
- SIGHAN Bake-off 2015: http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html  
- Wang271K: https://github.com/wdimmy/Automatic-Corpus-Generation
- Further pre-training corpus: https://github.com/brightmart/nlp_chinese_corpus
- Confusion set 1: http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html
- Confusion set 2: http://nlp.ee.ncu.edu.tw/resource/csc.html
- Confusion set 3: https://github.com/wdimmy/Automatic-Corpus-Generation


### Data Processing
The code and cleaned data are in the `data_process` directory.

You can also directly download the processed data from [this](https://drive.google.com/drive/folders/1dC09i57lobL91lEbpebDuUBS0fGz-LAk) and put them in the `data` directory. The `data` directory would look like this:
```
data
|- trainall.times2.pkl
|- test.sighan15.pkl
|- test.sighan15.lbl.tsv
|- test.sighan14.pkl
|- test.sighan14.lbl.tsv
|- test.sighan13.pkl
|- test.sighan13.lbl.tsv
```
