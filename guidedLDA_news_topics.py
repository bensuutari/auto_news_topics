
import numpy as np 
import guidedlda

X = guidedlda.datasets.load_data(guidedlda.datasets.NYT)

vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)
word2id = dict((v, idx) for idx, v in enumerate(vocab))

model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)

model.fit(X)

topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	print('Topic {}: {}'.format(i, ' '.join(topic_words)))