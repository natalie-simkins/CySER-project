# Author: Natalie!
# Purpose: reads an excel file and finds a list of most common topics, then lists the words associated with the topic, and then the weights of each topic.

#inspo for reading in from https://github.com/kapadias/medium-articles/blob/49af65a71fb4ba949d27bff086bdc0d4252fd577/natural-language-processing/topic-modeling/Evaluate%20Topic%20Models.ipynb
#reading in the csv info and assigning it a name in this program
import zipfile
import pandas as pd


with zipfile.ZipFile("dataset.xlsx.zip", "r") as zip_ref:
    zip_ref.extractall("temp")
    
dataset = pd.read_csv("temp/dataset2.csv")


#inspo from https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
#getting rid of extra columns/info
dataset = dataset.drop(columns=['uniqueProjectId','agencyCode','agencyName','currentUII','investmentName','investmentType','projectId','agencyProjectId','projectName','infrastructureManagementCategory','projectStatus','tmfInitiative','softwareProject','incrementalDevelopment','iterationFrequencyAmount','iterationFrequencyUnits','iterativeDescription','plannedStartDate','projectedStartDate','actualStartDate','plannedEndDate','projectedEndDate','actualEndDate','plannedCost','projectedCost','actualCost','scheduleVariance(days)','scheduleVariance(%)','scheduleVarianceColor','costVariance($M)','costVariance(%)','costVarianceColor','updatedTime'],axis=1).sample(100)
dataset.head()
#filter only Ys
dataset = dataset.loc[dataset['Legacy_Project?'] == 'Y']

#same inspo source as above
#removing punctuation/cases so we can process just words
import re
#no punctuation
dataset['projectGoal_processed'] = \
dataset['projectGoal'].map(lambda x: re.sub('[,.!?/]', '', str(x)))
#no uppercase
dataset['projectGoal_processed'] = \
projectGoalProcessed = dataset['projectGoal_processed'].map(lambda x: x.lower())
#printing out the first few rows to see if it worked
dataset['projectGoal_processed'].head()


#same inspo as above
#making a word cloud so that we have an idea of what the most common words will be
import sys
print(sys.executable)
from wordcloud import WordCloud
long_string=','.join(list(dataset['projectGoal_processed'].values))
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

#generating the wordcloud!
wordcloud.generate(long_string)
wordcloud.to_image().save("wordCloudNew.png")

#second attempt from different source
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#bigrams and trigrams
bigram = gensim.models.Phrases(projectGoalProcessed, min_count=20, threshold=100)
trigram = gensim.models.Phrases(bigram[projectGoalProcessed], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


#making the dictionary
projectGoalProcessed = projectGoalProcessed.fillna('').astype(str)
projectGoalProcessed = [simple_preprocess(str(d), deacc=True) for d in projectGoalProcessed]
#remove stop words
stop_words = stopwords.words('english')
projectGoalProcessed = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in projectGoalProcessed]
print(projectGoalProcessed[:1][0][:30])
id2word = corpora.Dictionary(projectGoalProcessed)

#making the corpus
corpus = [id2word.doc2bow(text) for text in projectGoalProcessed]

#building the actual LDA model
num_topics = 10
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100, update_every=1, chunksize=10, passes=10,alpha='symmetric', iterations=100, per_word_topics=True)

#printing
from pprint import pprint
pprint(lda_model.print_topics())

#visualizing the topics!
import pickle
import pyLDAvis
import pyLDAvis.gensim
import os


LDAvis_data_filepath = os.path.join('./results/prepared_'+str(num_topics))
#unleash the pickle
if 1 == 1:

    LDAvis_graph = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)


    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_graph, f)

#other pickle
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_graph = pickle.load(f)

#saving our pickles
pyLDAvis.save_html(LDAvis_graph, './results/prepared_'+str(num_topics) +'.html')
