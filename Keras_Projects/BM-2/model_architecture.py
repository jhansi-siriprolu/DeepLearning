##Model architecture
import keras
from keras.layers import InputLayer, Dense
from keras.models import Sequential
from keras.activations import linear, sigmoid, relu, leaky_relu
from keras.optimizers import Adam, RMSprop, SGD

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class model_architecture:
    def __init__(self, path):
        self.df_ohe = pd.read_csv(path)
    def __str__(self):
        return "Model Architecture Description"

    def __repr__(self):
        return "Model Architecture Representation"    
    def data_split(self):
        self.df = self.df_ohe.drop("Item_Identifier", axis = 1)
        print("Columns")
        print(self.df_ohe.columns)
        X = self.df.drop("Item_Outlet_Sales", axis = 1)
        y = self.df["Item_Outlet_Sales"]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, random_state=42, test_size=0.3)
        
        #defining neurons in each layer
        self.input_neurons = self.X_train.shape[1]
        self.neurons_hidden_layer_1 = 100
        self.neurons_hidden_layer_2 = 40
        self.neurons_hidden_layer_3 = 10
        self.output_neurons = 1
        
        self.model = Sequential()
        self.model.add(InputLayer(input_shape = (self.input_neurons,)))
        self.model.add(Dense(units = self.neurons_hidden_layer_1, activation=relu))
        self.model.add(Dense(units = self.neurons_hidden_layer_2, activation=sigmoid))
        self.model.add(Dense(units = self.neurons_hidden_layer_3, activation=relu))
        self.model.add(Dense(units=1, activation=linear))
        
        ### Compiling the model - Defining the loss fucntion and optimizer
        self.model.compile(loss=keras.losses.MeanSquaredLogarithmicError, optimizer="adam", metrics = ["msle"])
        
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs = 500)
    
    # def model_evaluation(self):
        self.pred_y_val = self.model.predict(self.X_val)
        mae = mean_absolute_error(self.y_val, self.pred_y_val)
        
        
        predicted = pd.DataFrame(self.pred_y_val, columns=["Item_Outlet_Sales"])
        result  = pd.concat([self.df_ohe["Item_Identifier"], self.df_ohe["Outlet_Identifier"], predicted["Item_Outlet_Sales"]], axis = 1)
        result.to_csv("final_result.csv",index = False)
        return mae
        
if __name__ == "__main__":
    df = model_architecture("preprocessed_train_data.csv")
    mae = df.data_split()
    print("Metric: ",mae)
    
    