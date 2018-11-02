import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning)

from gensim.models import Word2Vec, KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=10, n_components=2, init='pca', n_iter=600, random_state=23)
    new_values = tsne_model.fit_transform(tokens[:1000])

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    # matplotlib 한글 폰트 설정
    font_location = 'C:\WINDOWS\FONTS\MALGUN.TTF'
    font_name = fm.FontProperties(fname=font_location).get_name()
    plt.rc('font', family=font_name)


    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

#model = Word2Vec.load_word2vec_format('w2v.model')
model = KeyedVectors.load_word2vec_format('w2v.model')

# 트레이닝 완료되면 필요없는 메모리 unload
model.init_sims(replace=True)

#print(model.wv['폭행'])
#print(model.most_similar('폭행'))

tsne_plot(model)