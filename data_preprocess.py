"""
数据清洗、数据预处理
"""
import os
import pandas as pd
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation

"""
从csv文件读取数据，并将数据转化位DataFrame
"""
candidate_data = pd.read_csv('bisai/candidate_paper_for_wsdm2020.csv').fillna('')
train_data = pd.read_csv('bisai/train_release.csv').fillna('')
test_data = pd.read_csv('bisai/test.csv').fillna('')
"""
第一次运行需下载停用词表以及标点符号
利用nltk包中的stopwords进行操作，其中stopwords去除英文中的一些停用词
（i me my myself we our ours ourselves you you're）等
"""
# 第一次运行需下载停用词表以及标点符号
home_dir = os.environ['HOME']
nltk_dir = os.path.join(home_dir, 'nltk_data')
if not os.path.exists(nltk_dir):
    nltk.download('stopwords')
    nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def process_sentence(text):
    """
    对句子进行处理，对于一些分割符和单词进行空格处理
    对于分割之后的句子将其中的标点符号以及停用词去除
    """
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()
    # 去除标点符号以及停用词
    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]

    return ' '.join(tokens)


def process_pubmed(text):
    """
    TODO 处理PubMed摘要？
    Preprocess PubMed abstract. (https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PubMed/)
    1. tokenize sentences and words
    2. lowercase
    """
    ret = []
    has = False
    for sent in sent_tokenize(text):
        if '**##**' not in sent:
            continue
        has = True
        #         sent = sent.replace('[[**##**]]', '')
        #         text = ' '.join(word_tokenize(sent))
        ret.append(sent)
    if has:
        return ' '.join(ret)
    else:
        return 'NO_CITE'


# 所有文章数目，总共有838939篇
print(len(candidate_data))
# 将文献中不包含摘要的文章去掉
candidate_data = candidate_data[~candidate_data.abstract.str.contains('NO_CONTENT', regex=False)]
print(len(candidate_data))

# 对train数据进行处理，过滤description_text列包含NO_SITE的，
print(len(train_data))
train_data.description_text = train_data.description_text.apply(process_pubmed)
train_data = train_data[~train_data.description_text.str.contains('NO_CITE', regex=False)]
print(len(train_data))

# 对test数据进行处理，过滤description_text列包含NO_SITE
print(len(test_data))
test_data.description_text = test_data.description_text.apply(process_pubmed)
test_data = test_data[~test_data.description_text.str.contains('NO_CITE', regex=False)]
print(len(test_data))

candidate_data.to_hdf('cleaned2.h5', 'candidate_data')
train_data.to_hdf('cleaned2.h5', 'train_data')
test_data.to_hdf('cleaned2.h5', 'test_data')
