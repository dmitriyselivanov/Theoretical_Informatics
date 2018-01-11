from datetime import datetime

#NumPy (python3 -m pip install numpy)
import numpy as np

#scikit-learn (python3 pip install -U scikit-learn)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class predictor:
  input_data = []
  output_data = []
  unknown_data = []

  def seconds_from_timestamp(self, timestamp):
    transport_time = datetime.fromtimestamp(timestamp)
    time_in_seconds = float(transport_time.hour*3600 + transport_time.minute*60 + transport_time.second)
    return time_in_seconds

  def load_data(self, transport_file_path):
    transport_file = open('transport_data.csv')
    str_lines = transport_file.readlines()
    
    for each_line in str_lines:
      data_array = each_line.split(',') # array [longitude, latitude, serverUnixTime, transportUnixTime, routeNumber]
      route_number = data_array[4][0] # because data_array[4] looks like string '0\n'
      
      if route_number == '?':
        transport_time = self.seconds_from_timestamp(int(data_array[3]))
        element = [float(data_array[0]), float(data_array[1]), transport_time]
        self.unknown_data.append(element)
      elif route_number != '-':
        transport_time = self.seconds_from_timestamp(int(data_array[3]))
        element = [float(data_array[0]), float(data_array[1]), transport_time]
        self.input_data.append(element)
        self.output_data.append(int(route_number))

  def __init__(self, transport_file_path):
    self.load_data(transport_file_path)

  def predict(self, isDebug=False):
    X = np.array(self.input_data)
    y = np.array(self.output_data) 

    rfc = RandomForestClassifier(n_estimators = 2000, criterion = 'entropy', max_features = 3, oob_score = True, n_jobs=-1)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    if isDebug:
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 1) 
      rfc.fit(X_train, y_train)

      accuracy = rfc.score(X_test, y_test)
      print(accuracy)

    else:
      rfc.fit(X, y)

    recognized_data = rfc.predict(np.array(self.unknown_data))

    thefile = open('transport_output_forest.txt', 'w')
    for item in recognized_data:
      thefile.write("%s\n" % item)

prc = predictor("transport_data.csv")
prc.predict(isDebug=True)

