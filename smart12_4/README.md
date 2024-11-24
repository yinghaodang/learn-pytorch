Data from https://github.com/neulab/word-embeddings-for-nmt.git 4c5670b

I fixed the ted_reader.py line 90:
`with open(path_, 'r') as fp:` to `with open(path_, 'rb') as fp:`
and choose language as zh-cn.

**generate data**
```bash
cd smart12_4
python word-embeddings-for-nmt/ted_reader.py
```

Damn, I suspect the author did this on purpose. The code provided by the author is incomplete, deliberately omitting a critical file from the dataset. I can only make guesses based on the code they provided. The author's book also doesn't mention any details about data processing.

例子来自《深度学习原理与Pytorch实战》(集智俱乐部著)这本书的第12章，Transformer.ipynb是书本上的实现。
我本意是希望能做完整个训练流程的，但是数据处理方面配套资料有所缺失。仅实现了网络结构。
