from gensim.models import Word2Vec
import os
import pymysql
import nltk
import re
pattern_dot_num = re.compile(r'\d+[.]\d+')
#nltk.download('movie_reviews')
#from nltk.corpus import movie_reviews
from konlpy.tag import Mecab
mecab = Mecab()


def is_inc_hangul(s):
    for c in s:
        if c >= '가' and c <= '힣':
            return True
    return False



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
symbols = [',', '.', '\'', '‘','’', '"', '“', '”', '~', '`', '!', '?', '@', '$', '%', '/']

for i, row in enumerate(rows):
    if i > 10000:
        break
    tokenized = nltk.word_tokenize(row['sent_original'])
    print(tokenized)


    for j, token in enumerate(tokenized):
        for symbol in symbols:
            if symbol in token:
                # '%' token은 바로 앞 token에 붙여준다.
                if token == '%':
                    tokenized[j-1] = tokenized[j-1] + '%'
                    tokenized[j] = ''
                # 특수기호와 한글이 같이 있는 문자열이면 특수기호만 제거
                elif is_inc_hangul(token) == True:
                    tokenized[j] = tokenized[j].replace(symbol, '')
                # 소수점 숫자는 그냥 놔둔다.
                elif pattern_dot_num.findall(token):
                    break
                else:
                    # 특수기호 빈 문자열로 다 바꾸고 나중에 tokenized list에서 공백 모두 제거
                    tokenized[j] = ''
                break

    # 리스트에서 비어있는 원소 제거
    tokenized = ' '.join(tokenized).split()

    print(tokenized)
    print('\n')
    sentences.append(tokenized)
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




