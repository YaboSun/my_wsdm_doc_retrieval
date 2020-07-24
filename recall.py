import pandas as pd
import cupy
import cupyx
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp
from sklearn.feature_extraction.text import _document_frequency, CountVectorizer
from sklearn.utils.validation import check_is_fitted

candidate_data = pd.read_hdf('cleaned2.h5', 'candidate_data')
train_data = pd.read_hdf('cleaned2.h5', 'train_data')
test_data = pd.read_hdf('cleaned2.h5', 'test_data')

# 通过行来取行数据
test_data.iloc[9].to_dict()


class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    BM25可调参数
    use_idf:如果use_idf设置为True（默认值），则在转换期间会考虑反向文档频率（简单理解新鲜度）
    k1：控制词频结果在词频饱和度中的上升速度
    b：控制字段长的归一值
    BM25相关论文：Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """

    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        TODO 用来计算相似度？
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        """
        if not sp.isspmatrix(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape

            # _document_frequency计算某个词在文档中出现的次数
            # Count the number of non-zero values for each feature in sparse X.
            df = _document_frequency(X)
            # 逆文档频率
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_log = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
        return self

    def transform(self, X, copy=True):
        # 判断在X中是否存在dtype属性，并且需要X.dtype是np_float
        # dtype表示np多维数组中的数据类型属性
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            X = sp.csc_matrix(X, copy=copy)
        else:
            # 将整数或者二进制数转化为浮点数
            X = sp.csc_matrix(X, dtype=np.float64, copy=copy)
        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        dl = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]
        # In each row, repeat `dl` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # document_length = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(dl), sz)
        # 文档平均长度
        avgdl = np.average(dl)
        # 对非零元素计算其BM25分值
        # 由于绝大部分情况下，qi在Query中只会出现一次，即qfi=1,BM25算法原公式简化之后
        data = X.data * (self.k1 + 1) / (X.data + self.k1 * (1 - self.b + self.b * dl / avgdl))
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)
        if self.use_idf:
            check_is_fitted(self, '_idf_log', 'idf vector is not fitted')
            expected_n_features = self._idf_log.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model "
                                 "has been trained with n_features=%d" % (n_features, expected_n_features))
            X = X * self._idf_log
        return X


query = test_data['description_text']
doc = candidate_data['title'] + '' + \
      + candidate_data['journal'] + '' + \
      + candidate_data['keywords'] + '' + \
      + candidate_data['abstract']
all = pd.concat([doc, query])

cnt_vectorizer = CountVectorizer().fit(all)
cnt_all = cnt_vectorizer.transfrom(all)
bm25 = BM25Transformer().fit(cnt_all)

cnt_query = cnt_vectorizer.transfrom(query)
bm25_query = bm25.transform(cnt_query)

cnt_doc = cnt_vectorizer.transfrom(doc)
bm25_doc = bm25.transform(cnt_doc)


def call_bm25_rank():
    ans = []
    step = 150
    with cupy.cuda.Device(0):
        tf_bm25_doc = cupyx.scipy.sparse.csc_matrix(bm25_doc.T)

        def cal_a_query(start, step=10, topk=1000, tf_bm25_doc=None):
            tf_bm25_query = cupyx.scipy.sparse.csc_matrix(bm25_query[start:start + step, :])
            c = tf_bm25_query * (tf_bm25_doc)

            del tf_bm25_query
            cupy._default_memory_pool.free_all_blocks()
            c = cupy.argsort(-c.todense())[:, :topk]
            d = cupy.asnumpy(c)
            return d

        for i in range(0, len(test_data), step):
            ans.append(cal_a_query(i, step, 1000, tf_bm25_doc))
            if i % 3000 == 0:
                print(i)
    ret = np.vstack(ans)
    return ret


bm25_rank = call_bm25_rank()
