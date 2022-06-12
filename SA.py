import datetime
import pandas as pd
import streamlit as st
from PIL import Image
import snscrape.modules.twitter as sntwitter
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import altair as alt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
import httplib2
from bs4 import BeautifulSoup
import requests
# from scrapy.crawler import CrawlerProcess
from wordcloud import WordCloud, STOPWORDS
from streamlit_option_menu import option_menu

st.set_page_config(layout = 'wide',page_title='Analy✅s')
col1, col2 ,col3 = st.columns(3)
with col2:
    Logo = Image.open('Analytics.png')
    example = Image.open('download.jfif')
    st.image(Logo,width=450)

   

with st.sidebar:
    page = option_menu('Welcome',['Home', 'Pricing'],menu_icon='hdd-stack-fill',icons=['house','cash-coin'])
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This Sentiment Analysis [app](https://share.streamlit.io/giswqs/streamlit-template) is maintained by the [Doofenshmirtz Analytics Inc](https://wetlands.io).

#### Team19:

CEO - Sashank

CTO - Talha

CFO - Shoukath

Get in touch with us through:
+919361676498

Source code: <https://github.com/giswqs/streamlit-template>

"""
    )

def pipeline(brandname,start_date,end_date):
    
    query = "(#"+brandname+ ") lang:en until:"+end_date+" since:"+start_date+" -filter:replies"
    tweets = []
    limit = 100
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date,tweet.content])
            
    df = pd.DataFrame(tweets,columns=['Date','Tweet'])
    
    
    stop_words = stopwords.words('english') 
    punct = string.punctuation
    stemmer = PorterStemmer()
    
    
    
    X=df['Tweet']
    cleaned_data = []
    for i in range(len(X)):
        tweet=re.sub('[^a-zA-Z]',' ',X.iloc[i])
        tweet=tweet.lower().split()
        tweet=[stemmer.stem(word) for word in tweet if (word not in stop_words) and (word not in punct) ]
        tweet = ' '.join(tweet)
        cleaned_data.append(tweet)
        
    df1 = pd.DataFrame(cleaned_data,columns=['Cleand_Tweet'])
    df['Cleand_Tweet'] = df1['Cleand_Tweet']
    
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity
    

    def getPolarity(text):
        return TextBlob(text).sentiment.polarity
        
    
    df['Subjectivity']=df['Cleand_Tweet'].apply(getSubjectivity)
    df['Polarity']=df['Cleand_Tweet'].apply(getPolarity)
    
    
    allWords = ' '.join( [twts for twts in df['Cleand_Tweet']])
    wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119).generate(allWords)


    plt.imshow(wordCloud, interpolation = "bilinear")
    plt.axis('off')
    plt.savefig("words.jpg")
    
    
    def getAnalysis(score):
        if score < 0 :
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'
        
    df['Analysis'] = df['Polarity'].apply(getAnalysis)
    
    v= df['Analysis'].value_counts()
    
    df['Date'] = df['Date'].dt.date
    df = df[['Date','Polarity']].groupby('Date').sum()
    return v,df.reset_index()


def twitter(brand_name,st_date,ed_date):
    if brand_name == '':
        st.write('The Sentiment for Amazon is: \n <p style="color:red;text-align:center;font-size:35px">Negative</p>', unsafe_allow_html=True)
    else:
        polarity,df = pipeline(brandname=brand_name,start_date=st_date,end_date=ed_date)

        if polarity.idxmax() == 'Positive':
            Sentiment = 'Positive'
            color = 'green'
        elif polarity.idxmax() == 'Neutral':
            Sentiment = 'Neutral'
            color = 'orange'
        else:
            Sentiment = 'Negative'
            color = 'red'
        st.write('#### The Overall Sentiment for '+ brand_name +' is: <font style="color:'+color+'">'+Sentiment+'</font>', unsafe_allow_html=True)
        # ;text-align:center;font-size:35px

        v = alt.Chart(polarity.reset_index()).mark_arc().encode(
        alt.Color('index:N',scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'],range=['green', 'orange', 'red'])),
            theta = 'Analysis:Q', tooltip = ['Analysis','index']
        )
        st.write('#### Sentiment Distribution')
        st.altair_chart(v,use_container_width=True)

        time_series = alt.Chart(df).mark_line().encode(
        x = 'Date:T', y ='Polarity:Q',tooltip=['Date','Polarity'])
        st.write('#### Sentiment(Polarity) Score Trend')
        st.altair_chart(time_series,use_container_width=True)
        st.write('#### Keyword Frequency')
        example = Image.open('words.jpg')
    st.image(example)
    st.dataframe(df)


def news(search_product):
    # search_product = input("Search News ")
    url = 'https://www.ndtv.com/search?searchtext='+search_product
    response = requests.get(url)
    htmlcontent = response.content
    soup = BeautifulSoup(htmlcontent,"html.parser")
    news_title=[]
    news=[]
    headline = [x.get_text() for x in soup.find_all('div',attrs={'class':'src_itm-ttl'})]
    new= [x.get_text() for x in soup.find_all('div',attrs={'class':'src_itm-txt'})]
    news_title.append(headline)
    news.append(new)
    news_title = pd.DataFrame(news_title).T
    news = pd.DataFrame(news).T
    df_output = pd.concat([news_title,news], axis=1)
    df_output.columns=["News Title","News"]
    df_output.to_csv('news.csv')
    df =pd.read_csv('news.csv')
    pd.set_option("display.max_colwidth", -1)
    df = df.reset_index()
    df = df.drop(columns= "index")
    
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity
    
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity
    
    df['Subjectivity']=df['News'].apply(getSubjectivity)
    df['Polarity']=df['News'].apply(getPolarity)
    
    allWords = ' '.join( [twts for twts in df['News']])
    wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119).generate(allWords)


    plt.imshow(wordCloud, interpolation = "bilinear")
    plt.axis('off')
    plt.savefig("words.jpg")
    
    def getAnalysis(score):
        if score < 0 :
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Postive'
        
    df['Analysis'] = df['Polarity'].apply(getAnalysis)


    pos_df = df[df['Polarity']>0].sort_values(by= 'Polarity',ascending=False)['News Title'].drop_duplicates().head(5)
    st.write('#### Positive Headlines on ' + search_product)
    for headline in pos_df:
        st.write('  - ' + headline)
    
    neg_df = df[df['Polarity']<0].sort_values(by= 'Polarity',ascending=True)['News Title'].drop_duplicates().head(5)
    if len(neg_df)>0:
        st.write('#### Negative Headlines on ' + search_product)
        for headline in neg_df:
            st.write('  - ' + headline)

    st.write('#### Keyword Frequency')
    word = Image.open('words.jpg')
    st.image(word)
    # st.dataframe(df)

    return None


def flipkart(search_product):
    url = 'https://www.flipkart.com/search?q='+search_product
    http = httplib2.Http()
    response , content = http.request(url)
    links = []
    for link in BeautifulSoup(content).find_all('a',href=True):
        links.append(link['href'])
    newLink = []
    for link in links:
        newLink.append(link)  
        
    a = pd.DataFrame(newLink)
    new_url = 'https://www.flipkart.com'+a[0]
    new_url.drop(new_url.index[0:13],inplace=True)
    new_url.drop(new_url.index[20:],inplace=True)
    new1=new_url.tolist()
    
    products=[]
    prices=[]
    star=[]
    comments=[]
    for i in range(len(new1)):
        response = requests.get(new1[i])
        htmlcontent = response.content
        soup = BeautifulSoup(htmlcontent,"html.parser")
        product= [x.get_text() for x in soup.find_all('span',attrs={'class':'B_NuCI'})]
        price = [x.get_text() for x in soup.find_all('div',{'class':'_30jeq3 _16Jk6d'})]
        rating = [x.get_text() for x in soup.find_all('p', attrs={'class': '_2-N8zT'})]
        sta =  [x.get_text() for x in soup.find_all('div',{'class':'_2d4LTz'})]
        products.append(product)
        prices.append(price)
        comments.append(rating)
        star.append(sta)
    
    products = pd.DataFrame(products)
    prices = pd.DataFrame(prices)
    comments = pd.DataFrame(comments)
    star = pd.DataFrame(star)
        
        
    df_output = pd.concat([products,prices,star,comments], axis=1)

    df_output.columns=["PRODUCT","PRICE","OVERALL RATING","REVIEW 1","REVIEW 2","REVIEW 3","REVIEW 4","REVIEW 5","REVIEW 6","REVIEW 7","REVIEW 8","REVIEW 9","REVIEW 10"]

    df_new = df_output
    df_new.to_csv('FLIPKART.CSV')
    
    
    txt = df_output["REVIEW 1"].values
    wc = WordCloud(width=400, height=200, background_color='black').generate(str(txt))
    plt.figure(figsize=(20,10), facecolor='k', edgecolor='k')
    plt.imshow(wc, interpolation='bicubic') 
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig('flip_words.jpg')
    
    st.dataframe(df_new)
    
    return None

col1,col2,col3 = st.columns(3)



if page == 'Home':
    col2.markdown(
    """
    ### Product and Brand Sentiment Analysis

    """
    )
    brand_name = col2.text_input('Brand Name or Product Name', placeholder='Amazon')
    option = col2.selectbox('Option',['Twitter Analysis','News Analysis','Flipkart Analysis'])
    if option == 'Twitter Analysis':
        col1, col2,col3,col4 = st.columns([2,1,1,2])
        with col2:    
            st_date = st.date_input(
            "Please Enter the start date",
            datetime.date(2019, 1, 1)).strftime('%Y-%m-%d')
        with col3:
            ed_date = st.date_input(
            "Please Enter the end date",
                datetime.date(2022, 4, 25)).strftime('%Y-%m-%d')
        
    col1,col2,col3 = st.columns(3)
    with col2:
        if st.button('Submit'):
            if option == 'Twitter Analysis':
                twitter(brand_name,st_date,ed_date)

            elif option == 'News Analysis':
                news(brand_name)

            elif option == 'Flipkart Analysis':
                flipkart(brand_name)
elif page == 'Pricing':
    with col2:
        st.markdown('''
### Pricing Plan
##### Get our premium service at the cheapest prices
+ #### 3 month plan at Rs 969
+ #### 6 month plan at Rs 1769
+ #### Monthly add on at Rs 369
        
##### Contact us now to Activate your Subcription
        
        ''')

