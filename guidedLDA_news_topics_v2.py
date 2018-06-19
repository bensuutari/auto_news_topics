
import numpy as np 
import guidedlda
import pickle
import pandas as pd
import code
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
import glob
import code
import requests
from requests.auth import HTTPBasicAuth
import AUTH
from nltk.corpus import stopwords
import time
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

######LDA Settings############
no_features=1000
num_topics=5
disp_top_words=5
###############################


def runSKLearnLDA(doc_collection,no_topics,stop_words):
	print('Start SKLearnLDA...')
	tf_vec=CountVectorizer(max_df=0.95,min_df=2,max_features=no_features,stop_words=stop_words)
	termfreq=tf_vec.fit_transform(doc_collection)
	feature_names=tf_vec.get_feature_names()
	#Run LDA using scitkit learn
	print('Constructing LDA model...')
	startlda=time.time()
	ldamodel=LatentDirichletAllocation(n_components=no_topics, max_iter=10, learning_method='online', learning_offset=50.,random_state=0).fit(termfreq)#
	print('LDA Model Construction Took:'+str((time.time()-startlda)/60)+' minutes.')
	startldavecs=time.time()
	print('Constructing LDA vectors...')
	#ldavecs = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit_transform(termfreq,docidentifiers)	
	ldavecs=ldamodel.transform(termfreq)
	print('LDA Vector Construction Took:'+str((time.time()-startldavecs)/60)+' minutes.')
	print('Completed SKLearnLDA!')
	return termfreq,ldamodel,ldavecs,feature_names

def runGuidedLDA(doc_collection,no_topics,stop_words,seed_topics):
	print('Start SKLearnLDA...')
	tf_vec=CountVectorizer(max_df=0.95,min_df=2,max_features=no_features,stop_words=stop_words)
	termfreq=tf_vec.fit_transform(doc_collection)
	feature_names=tf_vec.get_feature_names()
	#Run LDA using scitkit learn
	print('Constructing GUIDED LDA model...')
	startlda=time.time()
	ldamodel=guidedlda.GuidedLDA(n_topics=no_topics, n_iter=100, random_state=7, refresh=20).fit(termfreq,seed_topics=seed_topics, seed_confidence=0.15)
	print('LDA Model Construction Took:'+str((time.time()-startlda)/60)+' minutes.')
	startldavecs=time.time()
	print('Constructing LDA vectors...')
	#ldavecs = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit_transform(termfreq,docidentifiers)	
	ldavecs=ldamodel.transform(termfreq)
	print('LDA Vector Construction Took:'+str((time.time()-startldavecs)/60)+' minutes.')
	print('Completed SKLearnLDA!')
	return termfreq,ldamodel,ldavecs,feature_names

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
def clean_text(inputstringlist):
	translator = str.maketrans('', '', string.punctuation)
	clean_text_list=list()
	for string_elem in inputstringlist:
		cleanstring=(string_elem.translate(translator)).lower()
		cleanstring=' '.join(cleanstring.split())
		clean_text_list.append(cleanstring)
	return clean_text_list

def score_perplexity(data,topic_nums):
	perplexity=list()
	numtopics=list()
	for num in topic_nums:
		numtopics.append(num)
		tf, ldamodel, ldavectors, tf_feature_names = runSKLearnLDA(data,num,stopwords.words('english'))		
		perplexity.append(ldamodel.perplexity(tf))
	plt.plot(numtopics,perplexity)
	plt.show()
	return numtopics,perplexity
r=requests.get("http://vault.elucd.com/news",auth=HTTPBasicAuth(AUTH.username, AUTH.password))
news_data=pd.DataFrame(r.json())
news_texts=news_data.text.tolist()
news_texts=clean_text(news_texts)


#external data from https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/#10diagnosemodelperformancewithperplexityandloglikelihood
#load_news_topics = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
#news_topics_external=clean_text(load_news_topics.content.tolist())
#tf_train_newsgroups, lda_train_newsgroups, ldavectors_train_newsgroups, tf_feature_names_newsgroups = runSKLearnLDA(news_topics_external,20,stopwords.words('english'))
display_topics(lda_train,tf_feature_names,disp_top_words)
code.interact(local=locals())
topic_no,perp=score_perplexity(news_topics_external,[1,3,5,10,20,30,100])
code.interact(local=locals())
topic_no,perp=score_perplexity(news_texts,[1,3,5,10,20,30,100])
code.interact(local=locals())
tf_train, lda_train, ldavectors_train, tf_feature_names = runSKLearnLDA(news_texts,num_topics,stopwords.words('english'))
display_topics(lda_train,tf_feature_names,disp_top_words)
print('{} total news articles used to construct LDA model'.format(len(news_texts)))
input()

seed_topic_list = [['drugs','marijuana','addict'],
					['homeless','safety'],
                    ['police','shooting','brutality','arrest'],
					['race','black','african','hispanic','latino','minority','discrimination','predjudice','racism','race'],
					['trump','politics','russia','mueller'],
					['weather','beach','vacation','food','dining']]
tf_train_guided, lda_train_guided, ldavectors_train_guided, tf_feature_names_guided = runGuidedLDA(news_texts,num_topics,stopwords.words('english'),seed_topic_list)
display_topics(lda_train_guided,tf_feature_names_guided,disp_top_words)

code.interact(local=locals())

'''
for title in news_data.keys():
	news_texts.append(news_data[title][0])
X = guidedlda.datasets.load_data(guidedlda.datasets.NYT)
vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)
word2id = dict((v, idx) for idx, v in enumerate(vocab))

model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)

model.fit(X)

topic_word = model.topic_word_
print(topic_word)
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# Guided LDA with seed topics.
seed_topic_list = [['game', 'team', 'win', 'player', 'season', 'second', 'victory'],
					['percent', 'company', 'market', 'price', 'sell', 'business', 'stock', 'share'],
                    ['music', 'write', 'art', 'book', 'world', 'film'],
					['political', 'government', 'leader', 'official', 'state', 'country', 'american','case', 'law', 'police', 'charge', 'officer', 'kill', 'arrest', 'lawyer']]

seed_topic_list = [['drugs','marijuana','addict'],
					['homeless','safety'],
                    ['police','shooting','brutality','arrest'],
					['race','black','african','hispanic','latino','minority','discrimination','predjudice','racism','race']]

print('STARTING GUIDED LDA WITH SEED TOPICS')
input()

model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
	for word in st:
		seed_topics[word2id[word]] = t_id
	model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)




loaddata=pickle.load(open(os.getcwd()+'/data/postdata2.pickle','rb'))
urls=loaddata.url
documents=list()
docsentiment=list()
docIDs=list()

for i,j,k in zip(loaddata.websitetext,loaddata.submissionID,loaddata.subreddit):
	if k is not 'moderatepolitics':
		if (type(i) is str):
			print('::::::::::::Doc Type: '+str(k))
			documents.append(i)
			docIDs.append(j)
			if k in ['conservative','republican','libertarian','the_congress']:
				docsentiment.append(0)#conservative
			elif k in ['liberal','democrats','politics','socialism']:
				docsentiment.append(1)#liberal
traindocs,trainlabels,testdocs,testlabels=train_test_split_data(documents,docsentiment)	
tf_train, lda_train, ldavectors_train = runSKLearnLDA(traindocs,trainlabels)
'''