import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import requests
from bs4 import BeautifulSoup

def get_site_content(url):
    ### Returnează conținutul site-ului sub forma unui șir de caractere ###
    response = requests.get(url)
    return response.text

def preprocess_site_content(html_content):
    ### Returnează conținutul prelucrat al site-ului, care conține doar textul site-ului și a eliminat orice caractere speciale ###
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    text = text.replace('\n', ' ').replace('\r', '')
    return text


nltk.download("stopwords") # descărcați setul de stopwords

# Încărcați datele
data = pd.read_csv("data.csv")

# Preprocesare date
stop_words = set(stopwords.words("english"))
corpus = []
for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['content'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in stop_words]
    review = ' '.join(review)
    corpus.append(review)

# Extrageți caracteristicile
vectorizer = CountVectorizer(max_features=1500)
X = vectorizer.fit_transform(corpus).toarray()
y = data.iloc[:, 1].values

# Divizați datele în set de antrenare și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Antrenați modelul
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluați modelul
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Preziceți un site nou
while True:
    new_site = input("Please enter the website: \n")
    try:
        if new_site.startswith(("https", "www")):
            new_site_content = get_site_content(new_site) # funcție care extrage conținutul site-ului
            new_site_content = preprocess_site_content(new_site_content) # funcție care preprocesează conținutul site-ului
            new_site_features = vectorizer.transform([new_site_content]).toarray()
            prediction = model.predict(new_site_features)
            if prediction[0] == 1:
                print("The", new_site, "is secure")
                break
            else:
                print(f"The, {new_site}, is not secure")
                break
        else:
            print("Please enter a valid link !")
    except requests.exceptions.MissingSchema:
        print("Please enter a valid link begin with http , https ")
    except requests.exceptions.ConnectionError:
        print("Site is down !")
print(model)
