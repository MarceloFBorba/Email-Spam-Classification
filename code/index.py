import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling  import SMOTE, BorderlineSMOTE, ADASYN

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unicodedata import normalize
from sklearn.metrics import classification_report, confusion_matrix

import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.linear_model    import LogisticRegression, SGDClassifier

import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from streamlit_elements import elements, mui, html

from PIL import Image

st.set_page_config(page_title='Email Spam Classifier', 
                   #page_icon='imagens/LogoFraudWatchdog.png',
                   layout='wide',
                   initial_sidebar_state='auto'
                   )
with st.sidebar:
    selected2 = option_menu("Menu",["Home", "Gráficos", "Sobre"], 
    icons=['house', 'database', 'graph-up', 'info-circle'], 
    menu_icon="menu-app", default_index=0,
    styles={
        "nav-link-selected": {"background-color": "#0378A6"}
    }
    )

# Lendo os dados

file_path ='https://www.dropbox.com/scl/fi/f8g7lukjhymkntut52707/email_spam.csv?rlkey=7rcjwi68138vqh4l03m011aip&dl=1'
df = pd.read_csv(file_path)



# texto de introdução
if (selected2 == "Home"):
    st.header('Analise de Emails Spam')
    col1, col2= st.columns([2,2])
    with col1:
        st.subheader('Introdução')
            
        st.markdown('<div style= "text-align: Justify;"> No contexto atual ofertado pelo crescimento exponencial da internet, a quantidade diária de emails transmitidos alcança a impressionante marca de 306.4 bilhões.\
            A projeção é ainda mais impactante, estimando que até 2025 esse valor alcance a cifra de 376.4 bilhões diários.\
            Contudo, esse crescimento vertiginoso não apenas contempla a esfera benéfica, mas também traz consigo o aumento proporcional de emails maliciosos.\
            Consequentemente, emerge uma urgência premente para o desenvolvimento de um sistema classificador de emails capaz de discernir e interceptar tais mensagens maliciosas antes mesmo que adentrem a caixa de entrada do usuário.\
            Nesse âmbito, instaura-se o presente projeto, com o intuito de investigar com minúcia o conjunto de dados em questão.\
            Almejamos identificar padrões intrínsecos que possam lançar luz sobre as similaridades e dissonâncias compartilhadas por emails tidos como normais e os que ostentam intenções maliciosas.</div>', unsafe_allow_html=True)
        
    with col2:
        image = Image.open("image/emails-spam.png")
        st.image(image, caption='Fonte: https://www.oberlo.com/blog/email-marketing-statistics', width=580) 

        
    with st.container():
        st.subheader('Dados usados')
        st.write('<div style= "text-align: Justify;"> Os fundamentais dados subjacentes a esta investigação foram generosamente disponibilizados pelo renomado portal Kaggle.com.\
            Esta plataforma renomada alberga uma diversidade exponencial de conjuntos de dados que enriquecem nossa pesquisa.\
            No cerne do presente conjunto de dados, encontramos colunas de vital relevância que catalisam nosso estudo.\
            Em particular, a coluna "type" emerge como um farol orientador, demarcando com precisão os emails como "spam" ou "not spam", conferindo uma categorização essencial para nossa análise.\
            No intricado mosaico de informações, a coluna "title" assume seu papel como peça-chave. Ela registra com acuidade os títulos singulares que dão identidade a cada mensagem.\
            A coluna "text", por sua vez, arquiva o corpo textual intrínseco dos emails, constituindo um tesouro de palavras e contextos cruciais para nossa pesquisa.\
            No empenho de nutrir nosso estudo com uma abrangência representativa, coletamos e exploramos um corpus robusto, englobando um total estimado de 84 emails.\
            Um meticuloso exame desvelou que cerca de 69% dessas comunicações são de caráter normal, enquanto os restantes 31% ostentam a classificação de "spam".\
            Assim, nossa análise transcende uma simples abordagem superficial e mergulha nas complexas nuances que caracterizam as mensagens eletrônicas contemporâneas.\
            Essa tapeçaria de dados selecionados meticulosamente do Kaggle.com foi diligentemente segmentada para as etapas cruciais de treinamento e teste.\
            Sob esse prisma, aproximadamente 80% dos dados foram alocados ao conjunto de treinamento, constituindo a base sólida sobre a qual erigimos nosso classificador.\
            Os restantes 20%, por sua vez, foram reservados para o conjunto de teste, representando uma amostra representativa que avalia a eficácia de nosso sistema em condições de simulação real.\
            Portanto, o ecossistema de dados fornecido pelo Kaggle.com converge com nossos esforços na busca por um classificador de emails apto a discernir entre o benigno e o malicioso,\
            contribuindo substancialmente para o desenvolvimento de soluções eficazes no enfrentamento dos desafios crescentes no cenário da comunicação eletrônica.</div>', unsafe_allow_html=True)
            
if (selected2 == "Gráficos"):
    st.subheader('Pagina graficos')
    graf1, graf2 = st.columns([2,2])
    gra3,graf4 = st.columns([2,2])
    
    with graf1:
        st.subheader('Dataframe original')
        st.write('Esse dataframe é o original, sem nenhuma alteração. \
        \n Como podemos ver, nosso conjunto de dados possui 3 colunas, o Title que representa o titlo do email,\
        o Text que representa o texto do email e a coluna Type que indica se\
        o email é ou não um Spam.')
        
        edited_df = st.data_editor(df)
    
    with graf2:
        st.subheader('Dataframe modificado')
        st.write('Esse Dataframe é resultado de após realizar o processamento de texto e a limpeza dos dados \
        retirando caracteres especiais e passando todas as letras minuscula e também aplicando o Stopwords para remover palavras Stopwords.')
        
        df.drop_duplicates()
        df.isnull().sum().any()
        df["spam"] = df["type"].apply(lambda x: 1 if x == "spam" else 0)
        df['message'] = df['title'] + df['text']
        df["message"] = df["message"].replace('\n','', regex=True)
        
        #aplicando a limpeza de dados
        def clean_text(df, message):
            df[f'{message}_cleaned'] = df[[f'{message}']]\
            .replace(regex=r'[!/,.@_?-]',value='')\
            .apply(lambda x: x.astype(str).str.lower())\
            .apply(lambda x:x.astype(str).str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))

        clean_text(df, 'message')
        df02 = df.drop(columns=['title', 'text', 'message'])
        
        # aplicando o stopwords
        
        nltk.download('stopwords')
        
        def stop_words_01( df02 , message_cleaned, message ):
            stop_words = stopwords.words('english')
            df02[message] = df02[message_cleaned].apply( lambda  x : ' '.join([word for word in x.split() if word not in (stop_words)]))
        
        stop_words_01(df02,'message_cleaned', 'message')
        df02 = df02.drop(columns=['message_cleaned'])
        
        df02
        
        
        
    with gra3:
        st.subheader('Gráfico 3')
        st.write('Gráfico da quantidade de emails por tipo')
        
        #Grafico da quantidade de emails por tipo
        df['type'].value_counts().plot(kind ="barh")
        
   
        
    with graf4:
        st.subheader('Gráfico 4')
        st.write('Gráfico 4')
        
        # Grafico de distribuicao de classes
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        f = sns.countplot(x = df02['spam'], palette="Blues_d")
        plt.xlabel('Target Variable')
        plt.ylabel('Counts of each class')
        plt.title('Class distribution (%)')
        for p in f.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{round(height/df02.shape[0], 2)*100} %', (x + width/2, y + height*1.01), ha='center')
        st.pyplot(fig)         

if (selected2 == "Sobre"):  
    st.write('Pagina sobre')  