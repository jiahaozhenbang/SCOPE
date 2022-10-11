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
- The code for cleaning data refers to [REALISE](https://github.com/DaDaMrX/ReaLiSe).

Recommend to directly download the cleaned data from [this](https://rec.ustc.edu.cn/share/b8470c00-4884-11ed-abb5-01b9f59aa971) and put them in the `data` directory. 

- process data to the training format. 

```
python data_process/get_train_data.py \
    --data_path data \
    --output_dir data
```

## Finetune

