import re
from sklearn.feature_extraction.text import CountVectorizer
class Parse():

    def __call__(self, text):
        text_split = text.split(" ")
        return [re.sub(r'[^\w\s]','',x) for x in text_split if len(x) > 3]

text = ["The quick brown fox jumped over the lazy dog.", "the cat is on the table", "the table is very hight 1025"]
# vectorizer = CountVectorizer()
vectorizer = CountVectorizer(max_df=2)
vectorizer.fit(text)
print(vectorizer.vocabulary_)
vector = vectorizer.transform(text)
print(vector.shape)
print(type(vector))
print(vector.toarray())