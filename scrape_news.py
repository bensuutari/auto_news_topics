import newspaper
from newspaper import Article
import pickle
import datetime

def scrape_local_news(news_sources):
	all_articles=dict()
	for news_outlet in news_sources.keys():
		all_news=newspaper.build(news_sources[news_outlet])
		for article_num,article in enumerate(all_news.articles):
			print(article.url)
			print(article_num)
			print(article.publish_date)
			article_content=Article(article.url)
			article_content.download()
			article_content.parse()
			all_articles[article_content.title]=[article_content.text,article_content.publish_date,article.url]
			'''
			with open('articles/'+news_outlet+'_'+article_content.title+'.txt','w') as article_out:
				article_out.write(article_content.text)
			'''
	todays_date=datetime.datetime.now()
	filename='{}_{}_{}_NY_local_news.pkl'.format(todays_date.year,todays_date.month,todays_date.day)
	pickle.dump(all_articles,open(filename,'wb'))


news=dict()
news['nbcny']='https://www.nbcnewyork.com/news/local/'
news['abc7ny']='http://abc7ny.com/'
news['cbslocalny']='http://newyork.cbslocal.com/category/news/ny-news/'

if __name__=='__main__':
	scrape_local_news(news)
