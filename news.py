from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

class predictor:
  input_data = []
  output_data = []
  unknown_data = []

  def load_data(self, news_train_path, news_test_path):
    news_file = open(news_train_path, 'r', encoding='utf-8')
    for each_line in news_file:
      raw_data = each_line.split('\t', maxsplit=1)
      self.output_data.append(raw_data[0])
      self.input_data.append(raw_data[1])
    
    news_test = open(news_test_path, 'r', encoding='utf-8')
    for each_line in news_test:
      self.unknown_data.append(each_line)

  def __init__(self, news_train_path, news_test_path):
    self.load_data(news_train_path, news_test_path)

  def predict(self):
      countVectorizer = CountVectorizer()
      transformer = TfidfTransformer()
      
      X = countVectorizer.fit_transform(self.input_data)
      X = transformer.fit_transform(X)
      
      text_classifier = LogisticRegression().fit(X, self.output_data)
     
      X_predict = countVectorizer.transform(self.unknown_data)
      X_predict = transformer.transform(X_predict)
      predicted_categories = text_classifier.predict(X_predict)
      
      thefile = open('news_output.txt', 'w')
      for category in predicted_categories:
        thefile.write("%s\n" % category)

prc = predictor("news_train.txt", "news_test.txt")
prc.predict()
