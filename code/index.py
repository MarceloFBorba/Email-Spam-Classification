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


st.header('Analise de Emails Spam')

# texto de introdução
if (selected2 == "Home"):
    
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
    st.write('Pagina graficos')

if (selected2 == "Sobre"):  
    st.write('Pagina sobre')  