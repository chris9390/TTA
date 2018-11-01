from gensim.models import Word2Vec
import os
import pymysql
import nltk
#nltk.download('movie_reviews')
#from nltk.corpus import movie_reviews

from konlpy.tag import Mecab
mecab = Mecab()

conn = pymysql.connect(host='163.239.169.54',
                       port=3306,
                       user='s20131533',
                       passwd='s20131533',
                       db='number_to_word',
                       charset='utf8',
                       cursorclass=pymysql.cursors.DictCursor)

c = conn.cursor()
sql = "SELECT sent_original FROM SentenceTable WHERE ArticleTable_article_id >= 620"
c.execute(sql)
rows = c.fetchall()

sentences = []
symbols = [',', '.', "'", '‘','’', '"', '“', '”', '~', '`', '!', '?', '@', '$', '%']

for i, row in enumerate(rows):
    if i > 10000:
        break
    print(nltk.word_tokenize(row['sent_original']))

    sentences.append(nltk.word_tokenize(row['sent_original']))
    #sentences.append(mecab.morphs(row['sent_original']))


#sentences = [list(s) for s in movie_reviews.sents()]



# Word2Vec 클래스 객체를 생성한다. 이 시점에 트레이닝이 이루어진다.
model = Word2Vec(sentences, size=100, window=2, iter=100, workers=os.cpu_count(), min_count=1, sg=1)

# 트레이닝 완료되면 필요없는 메모리 unload
model.init_sims(replace=True)

# 매번 model만들지 말고, 한번 만들고 나서 저장하고 로컬에서 model받아와서 실행하도록 코딩하자
#model.wv.save_word2vec_format('test.txt')


#print(model.wv['교사'])
print(model.most_similar('폭행', topn=30))

