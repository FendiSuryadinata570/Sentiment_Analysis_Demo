import streamlit as st
import pandas as pd
import pickle
import emoji
import re
import io
import base64
import csv
import time
import urllib
import string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from textblob import TextBlob
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
from sklearn import svm
from matplotlib.colors import Normalize
from numpy.random import rand
import matplotlib.cm as cm
from streamlit import caching
from datetime import datetime

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

loaded_model = pickle.load(open('Gabungan.pkl', 'rb'))
vectorizer = pickle.load(open('Gabungan_feature.pkl','rb'))
transformer = TfidfTransformer()
loaded_vec = TfidfVectorizer(decode_error="replace",vocabulary=vectorizer)

my_cmap = cm.get_cmap('jet')
my_norm = Normalize(vmin=0, vmax=8)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

@st.cache(show_spinner=False)
def get_text(raw_url):
    page = urlopen(raw_url)
    soup = BeautifulSoup(page)
    return ' '.join(map(lambda p:p.text, soup.find_all('p')))

def model_predict(text):
    X_sisa = transformer.fit_transform(loaded_vec.fit_transform([text])).toarray()
    predictions_test_SVM_sisa = loaded_model.predict(X_sisa)
    return str(predictions_test_SVM_sisa)

def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def input_choice_flow():
    input_choices_activities = ["Text","Upload File"]
    input_choice = st.sidebar.selectbox("Input Choices", input_choices_activities)
    if input_choice == "Text":
        caching.clear_cache()
        raw_text = st.text_area("Enter Text")
        if st.button("Analyze"):
            result = model_predict(raw_text)
            # st.write('Result: {}'.format(model_predict(raw_text)))
            if result == '[\'positive\']':
                st.info("Result :")
                custom_emoji = ':smile:'
                st.success(emoji.emojize(custom_emoji, use_aliases=True)+' Positive')
            elif result == '[\'neutral\']':
                st.info("Result :")
                custom_emoji = ':expressionless:'
                st.warning(emoji.emojize(custom_emoji, use_aliases=True)+' Neutral')
            elif result ==  '[\'negative\']':
                st.info("Result")
                custom_emoji = ':disappointed:'
                st.error(emoji.emojize(custom_emoji, use_aliases=True)+' Negative')


    elif input_choice == "Upload File":
        caching.clear_cache()
        if st.button('Download Sample Data'):
            df = pd.read_csv("test3.csv")
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  
            href = f'<a href="data:file/csv;base64,{b64}" download="sample.csv">Download csv file</a>'
            st.markdown(href, unsafe_allow_html=True)

        data = st.file_uploader("Upload Dataset", type=["csv","txt"])
        if data is not None:
            df = pd.read_csv(data, delimiter='\t')
            
            fig = go.Figure(data=[go.Table(header=dict(values=['Text']),
                 cells=dict(values=[df.Text]))
                     ])
            st.plotly_chart(fig)

            st.info("Sentiment Result: ")

            all_sentences = df['Text'].to_list()
            all_sentiment = [model_predict(x) for x in all_sentences]
            new_df = pd.DataFrame(zip(all_sentences, all_sentiment), columns=["Text","Sentiment"])

            a = len(df)
            a_pos = len(new_df[new_df['Sentiment']=='[\'positive\']'])
            a_neu = len(new_df[new_df['Sentiment']=='[\'neutral\']'])
            a_neg = len(new_df[new_df['Sentiment']=='[\'negative\']'])
            
            fig = go.Figure(data=[go.Table(header=dict(values=['Text', 'Sentiment']),
                 cells=dict(values=[new_df.Text, new_df.Sentiment]))
                     ])
            st.plotly_chart(fig)

            st.info("Download Result Here :")
            if st.button('Download Result as CSV'):
                tmp_download_link = download_link(new_df, 'result.csv', 'Click here to download your data!')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
            
            st.success('Positive Sentiment:')
            positive_df = new_df[new_df['Sentiment']=="[\'positive\']"]

            if a_pos > 0:
                fig = go.Figure(data=[go.Table(header=dict(values=['Text', 'Sentiment']),
                    cells=dict(values=[positive_df.Text, positive_df.Sentiment], fill_color="rgb(204,235,197)"))
                        ])
                st.plotly_chart(fig)

            else:
                st.write('No Positive Sentiment')

            st.warning('Neutral Sentiment:')
            neutral_df = new_df[new_df['Sentiment']=="[\'neutral\']"]
            if a_neu > 0:
                fig = go.Figure(data=[go.Table(header=dict(values=['Text', 'Sentiment']),
                    cells=dict(values=[neutral_df.Text, neutral_df.Sentiment], fill_color="rgb(255,242,174)"))
                        ])
                st.plotly_chart(fig)

            else:
                st.write('No Neutral Sentiment')

            st.error('Negative Sentiment:')
            negative_df = new_df[new_df['Sentiment']=="[\'negative\']"]

            if a_neg > 0:
                fig = go.Figure(data=[go.Table(header=dict(values=['Text', 'Sentiment']),
                    cells=dict(values=[negative_df.Text, negative_df.Sentiment], fill_color="rgb(249,123,114)"))
                        ])
                st.plotly_chart(fig)

            else:
                st.write('No Negative Sentiment')


            st.info("Result Stats: ")
            st.write('Length of Data: {}'.format(a))

            pos,neu,neg = st.beta_columns(3)
            pos.success('\+ Positive Data : {}'.format(a_pos))
            neu.warning('= Neutral Data : {}'.format(a_neu))
            neg.error('\- Negative Data : {}'.format(a_neg))

            sentiment_count = [a_pos, a_neu, a_neg]
            sentiment = ['Positive','Neutral','Negative']
 
            large = (3.18, 2.625)
            # normal = (2.385, 1.9688)
            fig, ax = plt.subplots(figsize=large)

            width = 0.3
            ind = np.arange(len(sentiment_count))
            ax.barh(sentiment, sentiment_count, width, color=my_cmap(my_norm(sentiment_count)))
            ax.set_yticklabels(sentiment, minor=False, fontsize=8)
            plt.title('Bar Chart', fontsize=8)
            for i, v in enumerate(sentiment_count):
                ax.text(v + .05, i-.05, str(v), color='black', fontweight='bold', fontsize=8) 
            st.pyplot()

def main():
    caching.clear_cache()
    st.title("Sentiment Analysis Demo")
    activities = ["Show Instructions","Sentiment", "Text Analysis of URL"]
    choice = st.sidebar.selectbox("Activities", activities)

    if choice == "Show Instructions":
        filename = 'instruct1.md'
        try:
            with open(filename) as input:
                st.subheader(input.read())
        except FileNotFoundError:
            st.error('File not found')
        st.sidebar.success('To continue select one of the activities.') 

    elif choice == "Sentiment":
        st.subheader("Sentiment Analysis")
        input_choice_flow()

    elif choice == "Text Analysis of URL":
        st.subheader("Analysis of Text from URL")

        raw_url = st.text_input("Enter URL")
        text_limit = st.slider("Length of Text to Preview", 50, 100)

        if st.button("Analyze"):
            result = get_text(raw_url)
            blob = TextBlob(result)
            len_of_full_text = len(result)
            len_of_short_text = round(len(result)/text_limit)
            st.info("Full Text Length: {}".format(len_of_full_text))
            st.info("Short Text Length: {}".format(len_of_short_text))
            st.write(result[:len_of_short_text])

            all_sentences = [sent for sent in blob.sentences]
            all_sentiment = [sent.sentiment.polarity for sent in blob.sentences]

            new_df = pd.DataFrame(zip(all_sentences, all_sentiment), columns=["Sentences","Sentiment"])
            st.dataframe(new_df)

            st.write(new_df.head(21).plot(kind='bar'))
            st.pyplot()

main()