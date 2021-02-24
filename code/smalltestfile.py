from timeseries_pytorch_simpleLSTM import testmod
from timeseries_pytorch_simpleLSTM import LSTM_manager
import pandas as pd

df = pd.read_csv("./synthetic_data/sinus_scenarios/noisy_sin_period126_missing20.csv")
data_name = 'noisy_sin'
data = df.filter([data_name])

# y_train, y_test = temporal_train_test_split(y, test_size=365)


s = LSTM_manager.LSTMHandler()
s.create_train_test_data(data = data, data_name = data_name)

s.create_trained_model(modelpath="timeseries_pytorch_simpleLSTM/testmodel.pt")

y_pred = s.make_predictions_from_model(modelpath="timeseries_pytorch_simpleLSTM/testmodel.pt")
