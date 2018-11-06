from gensim.models import Word2Vec
import os
import pymysql
import nltk
from konlpy.tag import Mecab
mecab = Mecab()
import logging
import re
pattern_dot_num = re.compile(r'\d+[.]\d+')



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
#sql = "SELECT sent_original FROM SentenceTable WHERE ArticleTable_article_id >= 620"
sql = "SELECT s.* FROM SentenceTable s, ArticleTable a WHERE a.article_sid1 = '사회' AND a.article_id = s.ArticleTable_article_id;"
c.execute(sql)
rows = c.fetchall()


mylogger = logging.getLogger('my')
mylogger.setLevel(logging.INFO)

#logfilename = './mylogs/mylog_morph_40000sents_5mincount.log'
#file_handler = logging.FileHandler(logfilename, mode='w')
#mylogger.addHandler(file_handler)
mylogger.info('sentences = 40000\nmin_count = 5\n')

sentences = []
symbols = [',', '.', '\'', '‘', '’', '"', '“', '”', '`', '!', '?', '@', '$', '%', '&', '/', '△', 'Δ', '▲', '▽', '▼', '◇', '◆', '=', '□', '■', '▷','▶','◁','◀' ]

for i, row in enumerate(rows):
    if i > 40000:
        break
    #tokenized = nltk.word_tokenize(row['sent_original'])
    tokenized = mecab.morphs(row['sent_original'])
    print('before : ' + str(tokenized))


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


    # 리스트에서 비어있는 원소 제거
    tokenized = ' '.join(tokenized).split()

    print('after  : ' + str(tokenized))
    print('\n')
    sentences.append(tokenized)


# word2vec 학습 전 토큰 개수 확인
token_dict_bef = {}
for tokens in sentences:
    for token in tokens:
        # 이미 토큰이 들어있으면 1증가
        if token in token_dict_bef:
            token_dict_bef[token] += 1
        # 처음 들어가는 토큰이면 1로 초기화
        else:
            token_dict_bef[token] = 1

print('before count : ' + str(len(token_dict_bef)))
mylogger.info('before token count : ' + str(len(token_dict_bef)))


# Word2Vec 클래스 객체를 생성한다. 이 시점에 학습이 이루어진다.
model = Word2Vec(sentences, size=300, window=2, iter=200, workers=os.cpu_count(), min_count=5, sg=1)



# word2vec 학습 후 토큰 개수 확인
token_dict_aft = {}
for vocab in model.wv.vocab:
    token_dict_aft[vocab] = model.wv.vocab[vocab].count

'''
import operator, copy
selected_token_list = []
token_dict_aft_asc = sorted(token_dict_aft.items(), key=operator.itemgetter(1), reverse=True)
for i, token in enumerate(token_dict_aft_asc):
    # 상위 10000개만 수집
    if i >= 10000:
        break
    selected_token_list.append(token[0])

model_copy = copy.deepcopy(model)
for vocab in model_copy.wv.vocab:
    if vocab not in selected_token_list:
        if vocab in model.wv.vocab:
            del model.wv.vocab[vocab]
'''

print('after count : ' + str(len(token_dict_aft)))
mylogger.info('after token count : ' + str(len(token_dict_aft)) + '\n')




# 트레이닝 완료되면 필요없는 메모리 unload
model.init_sims(replace=True)

# 매번 model만들지 말고, 한번 만들고 나서 저장하고 로컬에서 model받아와서 실행하도록 코딩하자
#model.wv.save_word2vec_format('w2v_10000sents.model')


'''
print(model.most_similar('폭행', topn=30))
mylogger.info('======= "폭행" 과 연관된 단어, 유사도 =======')
for each in model.most_similar('폭행', topn=30):
    mylogger.info(each)

mylogger.info('\n')

print(model.most_similar('채용', topn=30))
mylogger.info('======= "채용" 과 연관된 단어, 유사도 =======')
for each in model.most_similar('채용', topn=30):
    mylogger.info(each)


mylogger.info('\n')

print(model.most_similar('2019', topn=30))
mylogger.info('======= "2019" 와 연관된 단어, 유사도 =======')
for each in model.most_similar('2019', topn=30):
    mylogger.info(each)
'''


