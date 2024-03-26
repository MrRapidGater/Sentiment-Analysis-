#imported libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import emoji
import warnings
warnings.filterwarnings("ignore")


#reading csv file
data=pd.read_csv('semiclean_data.csv')

#converting date and time column to datetime format for later use
data['created_at'] = pd.to_datetime(data['created_at'])
data['user.created_at'] = pd.to_datetime(data['user.created_at'])


#new dataframe with selected columns
clean_data = data[['created_at', 'id', 'user.id', 'user.protected', 'user.verified', 'user.followers_count', 'user.friends_count', 'user.favourites_count', 'user.statuses_count', 'user.listed_count', 'user.created_at', 'user.location', 'quote_count', 'reply_count', 'retweet_count', 'favorite_count', 'text', 'truncated', 'extended_tweet.full_text', 'lang']]


#renaming columns
clean_data = clean_data.rename(columns={'created_at': 'Tweet date and time', 'id': 'Tweet ID', 'user.id': 'User ID', 'user.protected': 'Protected', 'user.verified': 'Verified', 'user.followers_count': 'Followers', 'user.friends_count': 'Friends', 'user.favourites_count': 'Favorites', 'user.statuses_count': 'Statuses', 'user.listed_count': 'List count', 'user.created_at': 'Account created at', 'user.location': 'User Location', 'quote_count': 'Quote count', 'reply_count': 'Reply count', 'retweet_count': 'Retweet count', 'favorite_count': 'Favorite count', 'lang': 'Language'})


#function for extracting full text using truncated column condition
tweets_text = []
for tweet in clean_data.index:
    if clean_data.iloc[tweet][17] == False:
        tweets_text.append(clean_data.iloc[tweet][16])
    else:
         tweets_text.append(clean_data.iloc[tweet][18])

             #STATS OF OUR DATA
print("Number of unique users: ", len(set(clean_data['User ID'])))
print("Total number of tweets: ",len(clean_data.index))

#counting number of tweeets that are replies
replies=0
for tweet in range(len(tweets_text)):
    if (re.search('^@', tweets_text[tweet])!=None):
        replies+=1       
print("Number of tweets that are replies: ", replies)

#counting number of tweets that are retweets
retweets=0
for tweet in range(len(tweets_text)):
    if (re.search('^RT @', tweets_text[tweet])!=None):
        retweets+=1
print("Number of tweets that are retweets: ", retweets)

#counting number of tweets that contain urls
urls=0
for tweet in range(len(tweets_text)):
    if (re.search('https://', tweets_text[tweet])!=None):
        urls=urls+1
print("Number of tweets that contain URLs: ", urls)



#stats for individual tweets
print("Total number of quotes: ",clean_data[clean_data.columns[12]].sum())
print("Total number of replies: ",clean_data[clean_data.columns[13]].sum()) 
print("Total number of retweets: ",clean_data[clean_data.columns[14]].sum())
print("Total number of favorites: ",clean_data[clean_data.columns[15]].sum())


#identifying languages of all tweets
clean_data['Language'].value_counts()

#counting tweets under each language
#all tweets other than english, urdu, ad hindi are grouped as 'others'
english=0
urdu=0
hindi=0
others=0
for i in tweets_df['Language'].index:
    if (tweets_df.iloc[i][2]=='en'):
        english=english+1
    elif (tweets_df.iloc[i][2]=='ur'):
        urdu=urdu+1
    elif (tweets_df.iloc[i][2]=='hi'):
        hindi=hindi+1
    else:
        others=others+1

#languages dataframe for plotting chart
languages = pd.DataFrame({'Language':['english', 'urdu', 'hindi', 'others'],
                    'Tweet count':[english, urdu, hindi, others]})

#barchart for language distribution of tweets
x=languages['Language']
y=languages['Tweet count']

p=sns.barplot(x,y, palette='mako')
p.axes.set_title("Language Distribution of Tweets",fontsize=20)
p.set_xlabel("Language",fontsize=15)
p.set_ylabel("Tweet Count",fontsize=15)
p.tick_params(labelsize=12)
for i in range(len(x)):
    plt.text(i, y[i]+1000, y[i], color='black', fontweight='bold', fontsize=12, ha='center', va='center')
plt.show()

#extracting hour and minute from time and date column and creating new columns with this info
hour=[]
hour = tweets_df['Tweet date and time'].dt.hour
minute=[]
minute = tweets_df['Tweet date and time'].dt.minute

#time dataframe for plotting horly tweet count
time = pd.DataFrame()
time['hour'] = hour
time['minute'] = minute

#variables for storing tweet counts
tweets_0600_0700 = 0
tweets_0700_0800 = 0
tweets_0800_0900 = 0
tweets_0900_1000 = 0
tweets_1000_1100 = 0
tweets_1100_1200 = 0

#calculating tweets in each hour
for i in time.index:
    if (time.iloc[i][0]==6):
        tweets_0600_0700+=1
    elif (time.iloc[i][0]==7):
        tweets_0700_0800+=1
    elif (time.iloc[i][0]==8):
        tweets_0800_0900+=1
    elif (time.iloc[i][0]==9):
        tweets_0900_1000+=1
    elif (time.iloc[i][0]==10):
        tweets_1000_1100+=1
    elif (time.iloc[i][0]==11):
        tweets_1100_1200+=1


#dataframe for storing hourly tweets for plotting
hourly_tweets = pd.DataFrame({'Time':['06:00-07:00', '07:00-08:00', '08:00-09:00', '09:00-10:00', '10:00-11:00', '11:00-12:00'],
                    'Tweet count':[tweets_0600_0700, tweets_0700_0800, tweets_0800_0900, tweets_0900_1000, tweets_1000_1100, tweets_1100_1200]})


#plotting barchart for hourly tweets
x=hourly_tweets['Time']
y=hourly_tweets['Tweet count']

sns.set(rc={'figure.figsize':(11,8)}, color_codes=True)
r = hourly_tweets['Tweet count'].argsort().argsort()
ax=sns.barplot(x='Time', y='Tweet count', data=hourly_tweets, palette=np.array(pal[::1])[r])
ax.set(ylim=(10000,20000))
ax.axes.set_title("Hourly Tweets Statistics",fontsize=20)
ax.set_xlabel("Time",fontsize=15)
ax.set_ylabel("Tweet Count",fontsize=15)
ax.tick_params(labelsize=12)
#adding value labels to each bar
for i in range(len(x)):
    plt.text(i, y[i]+100, y[i], color='black', fontweight='bold', fontsize=12, ha='center', va='center')
plt.show()


#tweets location stats for location having more than 1500 tweets
tweet_location = clean_data['User Location'].value_counts()
tweet_location = tweet_location[tweet_location >=1500]
tweet_location = pd.DataFrame(tweet_location)
tweet_location=tweet_location.reset_index()
tweet_location = tweet_location.rename(columns={'index': 'Location', 'User Location': 'Count'})


#plotting barplot for locations
x=tweet_location['Location']
y=tweet_location['Count']
pal=sns.color_palette("Blues")
pal.reverse()
ax=sns.barplot(y='Location', x='Count', data=tweet_location, palette=pal, orient='h')
ax.axes.set_title("Location of Users",fontsize=20)
ax.set_xlabel("Count",fontsize=15)
ax.set_ylabel("Location",fontsize=15)
#setting text values for each bar
for i, v in enumerate(y):
    ax.text(v + 100, i, str(v), color='black', fontweight='bold', fontsize=14, ha='left', va='center')



#new dataframe for users' stats
users=pd.DataFrame()
users['Tweet count']=clean_data['User ID'].value_counts(sort=False)
users = users.reset_index()
users = users.rename(columns={'index': 'User ID'})

#lists for appending text into users dataframe
creat=[]
prot=[]
ver=[]
foll=[]
frien=[]
fav=[]
stat=[]
lis=[]

#appending matching details to users dataframe
k=0
for i in range(0,57746):
    for j in range(k, 100000):
        if (users.iloc[i][0]==clean_data.iloc[j][2]):
            creat.append(clean_data.iloc[j][10])
            prot.append(clean_data.iloc[j][3])
            ver.append(clean_data.iloc[j][4])
            foll.append(clean_data.iloc[j][5])
            frien.append(clean_data.iloc[j][6])
            fav.append(clean_data.iloc[j][7])
            stat.append(clean_data.iloc[j][8])
            lis.append(clean_data.iloc[j][9])
            k=j
            break
users['Account created at']=creat         
users['Protected']=prot
users['Verified']=ver
users['Followers']=foll
users['Friends']=frien
users['Favorites']=fav
users['Statuses']=stat
users['List Count']=lis

#heatmap for checking correlation for users stats
heatdata = users[["Tweet count", "Account created at", "Followers", "Friends", "Favorites", "Statuses", "List Count"]]
sns.heatmap(heatdata.corr(), cmap="YlGnBu", annot=True)
plt.show()

#pie chart for protected users
fig = plt.figure(figsize=(7,7))
colors = ("pink", "skyblue")
wp = {'linewidth':2, 'edgecolor':'black'}
tags = users['Protected'].value_counts()
explode = (0.1, 0.1)
tags.plot(kind='pie', autopct= '%1.1f%%', colors=colors,  label=' ')
plt.title('Percentage of Protected Users')


#pie chart for verified users
fig = plt.figure(figsize=(7,7))
colors = ("pink", "skyblue")
wp = {'linewidth':2, 'edgecolor':'black'}
tags = users['Verified'].value_counts()
explode = (0.1, 0.1)
tags.plot(kind='pie', autopct= '%1.1f%%', colors=colors,  label=' ')
plt.title('Percentage of Verified Users')


#new dataframe for storing tweets
tweets_df = clean_data[['Tweet date and time', 'Tweet ID', 'Language']].copy()
tweets_df['Message'] = tweets_text
tweets_df

#filtering out english tweets and storing in new dataframe
tweets_df_eng = tweets_df[tweets_df['Language'] == 'en']
tweets_df_eng = tweets_df_eng.reset_index(drop=True)
tweets_df_eng


#function for cleaning tweet text
def cleanTweet(tweet):
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)
    tweet = re.sub(r"www.\S+", "", tweet)
    tweet = re.sub(r'@[A-Za-z0-9_:]+', '', tweet)
    tweet = re.sub(r'#+', '', tweet)
    tweet = re.sub(r'RT[\s]+', '', tweet)
    tweet = re.sub(r'\n', ' ', tweet)
    tweet = tweet.lower()
    tweet = emoji.demojize(tweet)
    tweet = tweet.replace(":"," ")
    tweet = ' '.join(tweet.split())
    text_tokens= word_tokenize(tweet)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

#all tweets are cleaned here
tweets_cleaned_text = []
for tweet in tweets_df_eng['Message']:
    tweets_cleaned_text.append(cleanTweet(tweet))


tweets_df_eng['Message'] = tweets_cleaned_text


   #SENTIMENT ANALYSIS
analyser = SentimentIntensityAnalyzer()

#calculating sentiment score and appending compound score into the tweets_df_eng dataframe
tweets_df_eng['Sentiment Score'] = [analyser.polarity_scores(x)['compound'] for x in tweets_df_eng['Message']]


#function for classifying tweets as either positive, neutral or negative
def sentiment(label):
    if label<0:
        return "negative"
    elif label==0:
        return "neutral"
    elif label>0:
        return "positive"

#calculating sentiment status of each tweet and appending into the dataframe
tweets_df_eng['Sentiment'] = tweets_df_eng['Sentiment Score'].apply(sentiment)


#pie chart for sentiment distribution
fig = plt.figure(figsize=(7,7))
colors = ("red", "yellowgreen", "gold")
wp = {'linewidth':2, 'edgecolor':'black'}
tags = tweets_df_eng['Sentiment'].value_counts()
explode = (0.1, 0.1, 0.1)
tags.plot(kind='pie', autopct= '%1.1f%%', shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, label=' ')
plt.title('Distribution of Sentiments')


#positive tweets dataframe for making wordcloud and for further analysis
pos_tweets = tweets_df_eng[tweets_df_eng.Sentiment == 'positive']
pos_tweets = pos_tweets.sort_values(['Sentiment Score'], ascending = False)
#positive tweets wordcloud
text = ' '.join([word for word in pos_tweets['Message']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=100, width=1000, height=500, background_color ='white').generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in positive tweets', fontsize=25, weight='bold')


#negative tweets dataframe for making wordcloud and for further analysis
neg_tweets = tweets_df_eng[tweets_df_eng.Sentiment == 'negative']
neg_tweets = neg_tweets.sort_values(['Sentiment Score'], ascending = True)
#negative tweets dataframe
text = ' '.join([word for word in neg_tweets['Message']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=100, width=1000, height=500, background_color ='white').generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in negative tweets', fontsize=25, weight='bold')

#resetting index of pos_tweets dataframe
pos_tweets = pos_tweets.reset_index(drop=True)
#resetting index of neg_tweets dataframe
neg_tweets = neg_tweets.reset_index(drop=True)

    #ASIA CUP TWEETS SENTIMENT ANALYSIS
#calculating positive tweets of asia cup
asiacup_pos=0
for i in pos_tweets.index:
    if (re.search('asiacup', pos_tweets.iloc[i][3])!=None or re.search('asia cup', pos_tweets.iloc[i][3])!=None):
        asiacup_pos+=1

#calculating negative tweets of asia cup
asiacup_neg=0
for i in neg_tweets.index:
    if (re.search('asiacup', neg_tweets.iloc[i][3])!=None or re.search('asia cup', neg_tweets.iloc[i][3])!=None):
        asiacup_neg+=1

#dataframe for positive asia cup tweets
asiacup = pd.DataFrame({'Sentiment':['positive', 'negative'],
                    'Count':[asiacup_pos, asiacup_neg]})

#pie chart of positve vs negative asia cup tweets
fig = plt.figure(figsize =(7, 7))
colors = ("yellowgreen", "red")
plt.pie([asiacup_pos,asiacup_neg],  autopct= '%1.1f%%', labels = ['positive', 'negative'], colors=colors, startangle=90)
plt.title('Sentiments of tweets containing "asiacup" or "asia cup"')
plt.show()


    #IMRAN KHAN TWEETS SENTIMENT ANALYSIS
#calculating positive tweets of imran khan
imrankhan_pos=0
for i in pos_tweets.index:
    if (re.search('imrankhan', pos_tweets.iloc[i][3])!=None or re.search('imran khan', pos_tweets.iloc[i][3])!=None):
        imrankhan_pos+=1

#calculating negative tweets of imran khan
imrankhan_neg=0
for i in neg_tweets.index:
    if (re.search('imrankhan', neg_tweets.iloc[i][3])!=None or re.search('imran khan', neg_tweets.iloc[i][3])!=None):
        imrankhan_neg+=1


#pie chart of positve vs negative imran khan tweets
fig = plt.figure(figsize =(7, 7))
colors = ("yellowgreen", "red")
plt.pie([imrankhan_pos,imrankhan_neg],  autopct= '%1.1f%%', labels = ['positive', 'negative'], colors=colors, startangle=90)
plt.title('Sentiments of tweets containing "imrankhan" or "imran khan"')
plt.show()


 #PAKISTAN TWEETS SENTIMENT ANALYSIS
#calculating positive tweets of pakistan
pakistan_pos=0
for i in pos_tweets.index:
    if (re.search('pakistan', pos_tweets.iloc[i][3])!=None):
        pakistan_pos+=1

#calculating negative tweets of paksitan
pakistan_neg=0
for i in neg_tweets.index:
    if (re.search('pakistan', neg_tweets.iloc[i][3])!=None):
        pakistan_neg+=1

#pie chart of positve vs negative pakistan tweets
fig = plt.figure(figsize =(7, 7))
colors = ("yellowgreen", "red")
plt.pie([pakistan_pos,pakistan_neg],  autopct= '%1.1f%%', labels = ['positive', 'negative'], colors=colors, startangle=90)
plt.title('Sentiments of tweets containing "pakistan"')
plt.show()


        #ARSHDEEP SINGH TWEETS SENTIMENT ANALYSIS
#calculating positive tweets of arshdeep
arshdeep_pos=0
for i in pos_tweets.index:
    if (re.search('arshdeep', pos_tweets.iloc[i][3])!=None):
        arshdeep_pos+=1

#calculating negative tweets of arshdeep
arshdeep_neg=0
for i in neg_tweets.index:
    if (re.search('arshdeep', neg_tweets.iloc[i][3])!=None):
        arshdeep_neg+=1


#pie chart of positve vs negative arshdeep tweeets
fig = plt.figure(figsize =(7, 7))
colors = ("yellowgreen", "red")
plt.pie([arshdeep_pos,arshdeep_neg],  autopct= '%1.1f%%', labels = ['positive', 'negative'], colors=colors, startangle=90)
plt.title('Sentiments of tweets containing "arshdeep"')
plt.show()



