# Forex Prediction
In the course of the last two decades the size and range of machine learning have grown enormously, and it is now widely recognized that they have many uses both in research and industries. One such use is the application of machine learning to finance-related studies. For instance, the foreign exchange market (Forex) is a global decentralized or over-the-counter market for the trading of currencies. In several companies, the main goal is to correctly predict the direction of future price movements of currency pairs. However, anticipating where the exchange rate is going on a consistent basis is far from easy, as dozens of different factors impact the forex market. In the past, investors and traders came up with a range of tools in trying to predict forex movements. In this study, I utilized different cutting-edge models to predict the direction of forex movement based on daily financial news headlines.

## Import Packages
```python
import warnings
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.offline import plot_mpl
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from pandas.core.common import flatten
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from xgboost import XGBClassifier
from collections import Counter
from collections import defaultdict

plt.style.use('seaborn-paper')
warnings.filterwarnings("ignore")
```

## Load News Data
I crawled the data from [Inshorts](https://inshorts.com/en/read) using web scraping technique. Web scaping is a technique of automating the extraction of data efficiently and effectively. I extract headlines and articles of financial news from by using python package, BeautifulSoup and Selenium. Inshorts is an app that has news stories in 60 words bites for majority of readers all over the world. It was set up by Azhar Iqubal, Anunay Pandey, and Deepit Purkayastha in 2013. For this study, there are three columns in the dataframe, namely date, news headline and news article. Total 2421 rows (from 2013-06-25 to 2020-03-08) can be seen in this dataframe. 

```python
news = pd.read_csv("news\indian_news_large.csv")
news["date"] = pd.to_datetime(news["news_date"])
news = news[news["news_category"] == "business"]
news = news[["date", "news_headline", "news_article"]]
news = news.groupby(['date'],as_index=False).agg({'news_headline': 'sum', "news_article": "sum"})
news.sort_values(by='date')
news.head()
```

||date|news_headline|news_article|
|---|---|---|---|
|0	|2013-06-25	|IIM-Kozhikode: 54.29% women in new batchGoogle...|IIM, Kozhikode's will have a record 54.29% of ...|
|1	|2013-06-26	|Telecom operators slashing data charges'Amul' ...|Smartphone owners in India are in for a treat ...|
|2	|2013-06-27	|Samsung shares hit a 9-month low|Samsung Electronics shares slumped 3% to a nin...|
|3	|2013-06-28	|Cabinet approves doubling of Gas pricesGoogle ...|Despite opposition, the Cabinet Committee on E...|
|4	|2013-06-29	|BlackBerry's share price dropped by 25%Petrol ...|BlackBerry's share price dropped by 25% in pre...|

## Load Forex Data
I downloaded the forex data from [Mecklai Financial](http://www.mecklai.com/Digital/MecklaiData/HistoryData_OHLC?AspxAutoDetectCookieSupport=1). After pre-processing, I get a new column "label", which means the differentiation between two days. If label is assigned to 1, it means this forex grew. Instead, the forex dropped. For this study, I used only GBP/USD price from the Macrotrends website. Macrotrends provides lots of foreign exchange data, such as EUR/USD, USD/JPY, USD/CNY, AUD/USD, EUR/GBP, USD/CHF, EUR/CHF, GBP/JPY and EUR/JPY. Moreover, they also illustrate interactive historical chart showing the daily forex price.

```python
# Get data from http://www.mecklai.com/Digital/MecklaiData/HistoryData_OHLC?AspxAutoDetectCookieSupport=1
fx = pd.read_csv("forex/Currency_Data_EURUSD.csv", index_col=False)
fx.columns = ["currecny", "date", "Open", "High", "Low", "Close"]
fx["date"] = pd.to_datetime(fx["date"])
fx.sort_values(by='date', inplace=True)
fx.reset_index(drop=True, inplace=True)
fx["label"] = fx["Close"].diff(periods=1)
fx.dropna(inplace=True)
fx.drop("currecny", axis=1, inplace=True)
fx["label"] = fx["label"].map(lambda x: 1 if float(x)>=0 else 0)
fx.head()
```

| |date|Open|High|Low|Close|label|
|---|---|---|---|---|---|---|
|1	|2007-01-10	|1.4232	|1.4238	|1.4140	|1.4155	|1|
|2	|2007-01-11	|1.4488	|1.4488	|1.4405	|1.4425	|1|
|3	|2007-02-08	|1.3669	|1.3705	|1.3652	|1.3704	|0|
|4	|2007-02-11	|1.4425	|1.4528	|1.4416	|1.4504	|1|
|5	|2007-03-08	|1.3703	|1.3819	|1.3684	|1.3774	|0|

## Combine News Data and Forex Data
```python
news_and_fx = pd.merge(news, fx, on=["date"])
news_and_fx.set_index('date', inplace=True)
news_and_fx["headline_len"] = news_and_fx["news_headline"].map(len)
news_and_fx["article_len"] = news_and_fx["news_article"].map(len)
print(Counter(news_and_fx["label"]))
news_and_fx
```

|	|news_headline	|news_article	|Open	|High	|Low	|Close	|label	|headline_len	|article_len|
|---|---|---|---|---|---|---|---|---|---|
|date ||||||||||									
|2013-06-25	|IIM-Kozhikode: 54.29% women in new batchGoogle...	|IIM, Kozhikode's will have a record 54.29% of ...	|1.5434	|1.5477	|1.5398	|1.5422	|0	|158	|1501|
|2013-06-26	|Telecom operators slashing data charges'Amul' ...	|Smartphone owners in India are in for a treat ...	|1.5422	|1.5440	|1.5298	|1.5314	|0	|139	|1274|
|2013-06-27	|Samsung shares hit a 9-month low	|Samsung Electronics shares slumped 3% to a nin...	|1.5314	|1.5346	|1.5202	|1.5259	|0	|32	|247|
|2013-06-28	|Cabinet approves doubling of Gas pricesGoogle ...	|Despite opposition, the Cabinet Committee on E...	|1.5259	|1.5279	|1.5166	|1.5213	|0	|199	|1962|
|2013-07-01	|Rupee worst among Asian currency in Q1Many B-s...	|The rupee lost as much as 8.6% in the April-Ju...	|1.6074	|1.6118	|1.6021	|1.6116	|1	|106	|1030|

## Visualisation
```python
news_and_fx.loc["2020-01-01":]["Close"].plot(figsize=(20, 8), title="USD/GBP", grid=True)
```
![img](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/news_and_fx_vis_gbpusd.png)

### Visualise Headline Sentence Length Distribution
```python
fig = px.histogram(news_and_fx, 
                   x="headline_len", 
                   color="label", 
                   marginal="rug", 
                   hover_data=news_and_fx.columns, 
                   color_discrete_sequence=px.colors.sequential.Sunsetdark)
fig.update_layout(
    title_text='Headline Sentence Length Distribution', 
    xaxis_title_text='Daily Sentence Length', 
    yaxis_title_text='Count'
)
fig.show()
# fig.write_image(file=r"C:\Users\YangWang\Desktop\currecy_prediction\image\Headline Sentence Length Distribution.png")
```
![img](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/Headline%20Sentence%20Length%20Distribution.png)

### Visualise Article Sentence Length Distribution
```python
fig = px.histogram(news_and_fx, 
                   x="article_len", 
                   color="label", 
                   marginal="rug", 
                   hover_data=news_and_fx.columns, 
                   color_discrete_sequence=px.colors.sequential.Sunsetdark)
fig.update_layout(
    title_text='Article Sentence Length Distribution', 
    xaxis_title_text='Daily Sentence Length', 
    yaxis_title_text='Count'
)
fig.show()
fig.write_image(file=r"C:\Users\YangWang\Desktop\currecy_prediction\image\Article Sentence Length Distribution.png")
```
![img](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/Article%20Sentence%20Length%20Distribution.png)

## Separate into Train/Test
```python
train = news_and_fx.loc[:"2019-03-07"]
test = news_and_fx.loc["2019-03-08":]
```

## Word Embedding

### Data Cleaning
1. Tokenize sentence into word using nltk.
2. Remove digital number.
3. Remove stopwords such as "the", "and", etc.

```python
def clean_text(text):
    tokens_list = word_tokenize(text)
    tokens_list = [word for word in tokens_list if word.isalpha()]
    tokens_list = [w for w in tokens_list if not w in stopwords.words('english')]
    return tokens_list
```

### Word2Vec
Build a vectorizer using [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf) in [gensim](https://radimrehurek.com/gensim/models/word2vec.html). Word2Vec is a model architecture for computing continuous vector representations of words. In this study, I use pre-trained Word2Vec from the officail Word2Vec [website](https://code.google.com/archive/p/word2vec/). Google published this pre-trained vectors on part of Google News dataset (about 100 billion words). The model contaings 300-dimensional vectors for 3 million words and phrases. The phrases were obtained using a simple data-driven approach described in Tomas Mikolov [paper](http://arxiv.org/pdf/1301.3781.pdf).

```python
class Word2VecVectorizer:
    def __init__(self):
        # Load in pretrained word vectors from https://github.com/Kyubyong/wordvectors
        print("Loading in word vectors...")
        self.w2v = KeyedVectors.load_word2vec_format(
            r"F:\embedding_file\GoogleNews-vectors-negative300.bin", binary=True)
        self.word2vec = {w: vec for w, vec in zip(self.w2v.index2word, self.w2v.vectors)}
        print("Finished loading in word vectors.")
        
    def fit(self, data):
        pass
    
    def transform(self, data):
        # Dimension of feature
        self.D = self.w2v.vector_size
        
        # Convert sentences using bag of word
        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            # First, remove stopwords and emoji. Second, tokenise sentence using mecab.
            tokens = clean_text(sentence)
            vecs = []
            m = 0
            for word in tokens:
                try: 
                    vec = self.w2v.wv.get_vector(word)
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    pass
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print("Number of samples with no words found: %s / %s" % (emptycount, len(data)))
        return X
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
```

### Word2Vec with TF-IDF
The bag-of-words model is a simplifying representation used in natural language processing and information retrieval (IR). In this model, a text (such as a sentence or a document) is represented as the bag of its words, disregarding grammar and even word order but keeping multiplicity. However, not all words equally represent the meaning of a particular sentence using `Word2VecVectorizer`. Therefore, I specify weights for the words calculated for instance using TF-IDF.

```python
# Reference from https://www.kaggle.com/mohanamurali/bgow-tf-idf-lr-w2v-lgb-bayesopt
class TfidfWord2VecVectorizer:
    def __init__(self):
        # Load in pretrained word vectors from https://github.com/Kyubyong/wordvectors
        print("Loading in word vectors...")
        self.w2v = KeyedVectors.load_word2vec_format(
            r"F:\embedding_file\GoogleNews-vectors-negative300.bin", binary=True)
        self.word2vec = {w: vec for w, vec in zip(self.w2v.index2word, self.w2v.vectors)}
        self.word2weight = None
        self.dim = self.w2v.wv.vector_size
        print("Finished loading in word vectors.")
        
    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
```

### Evaluate Model Performance
Evaluate the modle performance using accuracy, f1 score, roc-auc score, matthews correlation coefficient, and confusion matrix.
```python
def performance(model, x_train, y_train, x_test, y_test):
    print("test accuracy: ", round(model.score(x_test, y_test), 4))
    print("f-beta score: ", round(f1_score(y_test, model.predict(x_test)), 4))
    print("roc auc score: ", round(roc_auc_score(y_test, model.predict(x_test)), 4))
    print("matthews corrcoef: ", round(matthews_corrcoef(y_test, model.predict(x_test)), 4))
    print("confusion matrix: \n", confusion_matrix(y_test, model.predict(x_test)))
```

## Transform Data
Transform sentence into vectors. There are two vectorizer, namely `Word2VecVectorizer` and `TfidfWord2VecVectorizer`.
1. If using `TfidfWord2VecVectorizer`, then set it to True. Instead, set it to False.
2. If using news headline as training data, then set it to True. Instead, set it to False.

```python
weighted = True
using_headline = True

if weighted:
    if using_headline:
        print("Using TfidfWord2VecVectorizer.\n")
        vectorizer = TfidfWord2VecVectorizer()
        vectorizer.fit(train["news_headline"], train.label)
        x_train = vectorizer.transform(train.news_headline)
        y_train = train.label
        x_test = vectorizer.transform(test.news_headline)
        y_test = test.label
        print("\n# of train data: {}\n# of features: {}\n".format(x_train.shape[0], x_train.shape[1]))
        print("\n# of test data: {}\n# of features: {}\n".format(x_test.shape[0], x_test.shape[1]))
        print("Training data: \n", x_train)
    else:
        print("Using TfidfWord2VecVectorizer.\n")
        vectorizer = TfidfWord2VecVectorizer()
        vectorizer.fit(train["news_article"], train.label)
        x_train = vectorizer.transform(train.news_article)
        y_train = train.label
        x_test = vectorizer.transform(test.news_article)
        y_test = test.label
        print("\n# of train data: {}\n# of features: {}\n".format(x_train.shape[0], x_train.shape[1]))
        print("\n# of test data: {}\n# of features: {}\n".format(x_test.shape[0], x_test.shape[1]))
        print("Training data: \n", x_train)
else: 
    if using_headline:
        print("Using Word2VecVectorizer.\n")
        vectorizer = Word2VecVectorizer()
        x_train = vectorizer.fit_transform(train.news_headline)
        y_train = train.label
        x_test = vectorizer.fit_transform(test.news_headline)
        y_test = test.label
        print("\n# of train data: {}\n# of features: {}\n".format(x_train.shape[0], x_train.shape[1]))
        print("\n# of test data: {}\n# of features: {}\n".format(x_test.shape[0], x_test.shape[1]))
        print("Training data: \n", x_train)
    else: 
        print("Using Word2VecVectorizer.\n")
        vectorizer = Word2VecVectorizer()
        x_train = vectorizer.fit_transform(train.news_article)
        y_train = train.label
        x_test = vectorizer.fit_transform(test.news_article)
        y_test = test.label
        print("\n# of train data: {}\n# of features: {}\n".format(x_train.shape[0], x_train.shape[1]))
        print("\n# of test data: {}\n# of features: {}\n".format(x_test.shape[0], x_test.shape[1]))
        print("Training data: \n", x_train)
```

## Data Training

### XGBoost with RandomizedSearchCV
XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. 

```python
xgb = XGBClassifier(
    learning_rate=0.01,  
    n_estimators=400, 
    random_state=17, 
    slient = 0)

params = { 
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [5, 10, 15, 20]
}

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=17)

random_search = RandomizedSearchCV(
    xgb, 
    param_distributions=params, 
    n_iter=param_comb, 
    scoring='roc_auc', 
    n_jobs=6, 
    cv=skf.split(x_train, y_train), 
    verbose=2, 
    random_state=17)
random_search.fit(x_train, y_train)
xgb = random_search.best_estimator_
performance(xgb, x_train, y_train, x_test, y_test)
```

```console
test accuracy:  0.4978
f-beta score:  0.4887
roc auc score:  0.4998
matthews corrcoef:  -0.0003
confusion matrix: 
 [[58 64]
 [49 54]]
```

### Extra Trees with RandomizedSearchCV
Adding one further step of randomization yields extremely randomized trees, or ExtraTrees. While similar to ordinary random forests in that they are an ensemble of individual trees, there are two main differences: first, each tree is trained using the whole learning sample (rather than a bootstrap sample), and second, the top-down splitting in the tree learner is randomized.

```python
etc = ExtraTreesClassifier(
    criterion='gini', 
    random_state=17)

params = { 
    'n_estimators': [100, 200, 300]
}

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=17)

random_search = RandomizedSearchCV(
    etc, 
    param_distributions=params, 
    n_iter=param_comb, 
    scoring='roc_auc', 
    n_jobs=6, 
    cv=skf.split(x_train, y_train), 
    verbose=2, 
    random_state=17)
random_search.fit(x_train, y_train)
etc = random_search.best_estimator_
performance(etc, x_train, y_train, x_test, y_test)
```

```console
test accuracy:  0.4978
f-beta score:  0.4978
roc auc score:  0.5014
matthews corrcoef:  0.0027
confusion matrix: 
 [[56 66]
 [47 56]]
```

### Random Forest with RandomizedSearchCV
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

```python
rfc = RandomForestClassifier(
    n_estimators=200, 
    random_state=17)

params = { 
    'n_estimators': [100, 200, 300]
}

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=17)

random_search = RandomizedSearchCV(
    rfc, 
    param_distributions=params, 
    n_iter=param_comb, 
    scoring='roc_auc', 
    n_jobs=6, 
    cv=skf.split(x_train, y_train), 
    verbose=2, 
    random_state=17)
random_search.fit(x_train, y_train)
rfc = random_search.best_estimator_
performance(rfc, x_train, y_train, x_test, y_test)
```

```console
test accuracy:  0.5556
f-beta score:  0.5726
roc auc score:  0.5629
matthews corrcoef:  0.1272
confusion matrix: 
 [[58 64]
 [36 67]]
```

### Decision Tree with RandomizedSearchCV
Decision tree learning is one of the predictive modelling approaches used in statistics, data mining and machine learning. It uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves).

```python
dtc = DecisionTreeClassifier(
    criterion='gini', 
    random_state=17)

params = { 
    'max_depth': [5, 10, 15, 20, 25, 30]
}

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=17)

random_search = RandomizedSearchCV(
    dtc, 
    param_distributions=params, 
    n_iter=param_comb, 
    scoring='roc_auc', 
    n_jobs=6, 
    cv=skf.split(x_train, y_train), 
    verbose=2, 
    random_state=17)
random_search.fit(x_train, y_train)
dtc = random_search.best_estimator_
performance(dtc, x_train, y_train, x_test, y_test)
```

```console
test accuracy:  0.4756
f-beta score:  0.487
roc auc score:  0.4809
matthews corrcoef:  -0.0385
confusion matrix: 
 [[51 71]
 [47 56]]
```

### KNN with RandomizedSearchCV
In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation.

```python
knn = KNeighborsClassifier(
    metric="minkowski")

params = { 
    'n_neighbors': [6, 8, 10, 12], 
    "leaf_size": [25, 30, 35]
}

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=17)

random_search = RandomizedSearchCV(
    knn, 
    param_distributions=params, 
    n_iter=param_comb, 
    scoring='roc_auc', 
    n_jobs=6, 
    cv=skf.split(x_train, y_train), 
    verbose=2, 
    random_state=17)
random_search.fit(x_train, y_train)
knn = random_search.best_estimator_
performance(knn, x_train, y_train, x_test, y_test)
```

```console
test accuracy:  0.5333
f-beta score:  0.4776
roc auc score:  0.5281
matthews corrcoef:  0.0565
confusion matrix: 
 [[72 50]
 [55 48]]
```

### AdaBoost with RandomizedSearchCV
AdaBoost, short for Adaptive Boosting, is a machine learning meta-algorithm formulated by Yoav Freund and Robert Schapire, who won the 2003 Gödel Prize for their work. It can be used in conjunction with many other types of learning algorithms to improve performance. The output of the other learning algorithms ('weak learners') is combined into a weighted sum that represents the final output of the boosted classifier. AdaBoost is adaptive in the sense that subsequent weak learners are tweaked in favor of those instances misclassified by previous classifiers.

```python
ada = AdaBoostClassifier(
    random_state=17)

params = { 
    'n_estimators': [25, 50, 75, 100], 
    "learning_rate": [1, 0.1, 0.01, 0.001]
}

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=17)

random_search = RandomizedSearchCV(
    ada, 
    param_distributions=params, 
    n_iter=param_comb, 
    scoring='roc_auc', 
    n_jobs=6, 
    cv=skf.split(x_train, y_train), 
    verbose=2, 
    random_state=17)
random_search.fit(x_train, y_train)
ada = random_search.best_estimator_
performance(ada, x_train, y_train, x_test, y_test)
```

```console
test accuracy:  0.4889
f-beta score:  0.4978
roc auc score:  0.4939
matthews corrcoef:  -0.0122
confusion matrix: 
 [[53 69]
 [46 57]]
```

## Backtesting
The code below is reference from [QUANTSTART](https://www.quantstart.com/articles/Research-Backtesting-Environments-in-Python-with-pandas/).

```python
class Strategy(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_signals(self):
        raise NotImplementedError("Should implement generate_signals()!")
        
class Portfolio(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_positions(self):
        raise NotImplementedError("Should implement generate_positions()!")

    @abstractmethod
    def backtest_portfolio(self):
        raise NotImplementedError("Should implement backtest_portfolio()!")

        
class RandomForecastingStrategy(Strategy):   
    
    def __init__(self, symbol, bars):
        self.symbol = symbol
        self.bars = bars

    def generate_signals(self):
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = np.sign(np.random.randn(len(signals)))
        return signals
    
class MarketIntradayPortfolio(Portfolio):
    
    def __init__(self, symbol, bars, signals, initial_capital=100000, trading_sum=100):
        self.symbol = symbol        
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.trading_sum = float(trading_sum)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.symbol] = self.trading_sum*self.signals['signal']
        return positions
                    
    def backtest_portfolio(self):
        portfolio = pd.DataFrame(index=self.positions.index)
        pos_diff = self.positions.diff()
        
        
        portfolio['price_diff'] = self.bars['Close']-self.bars['Open']
        portfolio['profit'] = self.positions[self.symbol] * portfolio['price_diff']

        portfolio['total'] = self.initial_capital + portfolio['profit'].cumsum()
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio
    
class MachineLearningForecastingStrategy(Strategy):   
    
    def __init__(self, symbol, bars, pred):
        self.symbol = symbol
        self.bars = bars

    def generate_signals(self, pred):
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = pred
        return signals
```

## Create Strategy
```python
def model_backtesting(model, x_test, test, model_name, currency_name, initial_capital=100000, trading_sum=10000):
    pred = model.predict(x_test)
    pred = pred.tolist()
    pred = [1 if p == 1 else -1 for p in pred]
    
    test['Close'] = test['Close'].shift(-1)
    rfs = MachineLearningForecastingStrategy(currency_name, test, pred)
    signals = rfs.generate_signals(pred)
    portfolio = MarketIntradayPortfolio(currency_name, test, signals, initial_capital, trading_sum)
    returns = portfolio.backtest_portfolio()
    
    returns['signal'] = signals
    our_pct_growth = returns['total'].pct_change().cumsum()
    benchmark_ptc_growth = test['Close'].pct_change().cumsum()
    
    plt.figure(figsize=(20, 8))
    plt.subplot(2, 1, 1)
    plt.plot(returns['total'])
    plt.title("Total Return on {} using {}".format(currency_name, model_name))
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(20, 8))
    plt.subplot(2, 1, 2)
    plt.plot(our_pct_growth, label = 'ML long/short strategy', linewidth=2)
    plt.plot(benchmark_ptc_growth, linestyle = '--', label = 'Buy and hold strategy', linewidth=2)
    plt.title("ML Strategy V.S. Buy and Hold Strategy\nModel: {}\nCurrecy: {}".format(model_name, currency_name))
    plt.legend()
    plt.grid()
    plt.show()
```

### Start Backtesting

#### Backtesting using XGBoost
![XGBoost](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/xgboost_backtesting_1.png)
![XGBoost](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/xgboost_backtesting_2.png)

#### Backtesting using Extra Tree
![Extra Tree](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/extratrees_backtesting_1.png)
![Extra Tree](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/extratrees_backtesting_2.png)

#### Backtesting using Random Forest
![Random Forest](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/randomforest_backtesting_1.png)
![Random Forest](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/randomforest_backtesting_2.png)

#### Backtesting using Decision Tree
![Decision Tree](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/decisiontrees_backtesting_1.png)
![Decision Tree](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/decisiontrees_backtesting_2.png)

#### Backtesting using KNN
![KNN](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/knn_backtesting_1.png)
![KNN](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/knn_backtesting_2.png)

#### Backtesting using AdaBoost
![AdaBoost](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/adaboost_backtesting_1.png)
![AdaBoost](https://github.com/penguinwang96825/Forex-Prediction/blob/master/image/adaboost_backtesting_2.png)
