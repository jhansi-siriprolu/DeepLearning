import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

##Model architecture
import keras
from keras.layers import InputLayer, Dense
from keras.models import Sequential
from keras.activations import linear, sigmoid, relu, leaky_relu
from keras.optimizers import Adam, RMSprop, SGD

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class Preprocessing:
    def __init__(self,path, test = False):
        self.df2 = pd.read_csv(path)
        self.test = test

    def pre_processed_data(self):
        self.cat_cols = list(self.df2.select_dtypes("O").columns)
        #before that check if there is any category column that is a unique identifier
        self.remove_cols = []
        for col in self.cat_cols:
            if len(self.df2[col].unique()) > 1000:
                self.remove_cols.append(col)
        self.df = self.df2.drop(self.remove_cols, axis = 1)

    #categorical columns
        self.cat_cols = list(self.df.select_dtypes("O").columns)
    # def numeric_columns(self):
        self.num_cols =  [col for col in self.df.columns if col not in self.cat_cols]
        self.discrete_cols = [col for col in self.num_cols if len(self.df[col].unique()) < 20]
        self.continous_cols = [col for col in self.num_cols if col not in self.discrete_cols]

        
    # def impute_columns(self):
        self.discrete_and_categorical_cols = []
        self.discrete_and_categorical_cols.extend(self.cat_cols)
        self.discrete_and_categorical_cols.extend(self.discrete_cols)
        
        #Impute discrete and categorical columns with mode
        for col in self.discrete_and_categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        #Imputing continous numeric columns with mean
        for col in self.continous_cols:
            if self.df[col].isnull().sum()>0:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
                            
    # def categorical_numeric(self):
        self.nominal_cat_cols = ["Item_Type", "Outlet_Type","Outlet_Identifier","Outlet_Location_Type","Outlet_Type"]
        self.ordinal_cat_cols = ["Item_Fat_Content", "Outlet_Size"]
        
        Item_Fat_Content_mapping = {"Low Fat" : 0, "low fat" : 0, "LF" : 0,
                            "Regular" : 1, "reg" : 1}

        Outlet_Size_mapping = {"Small": 0, "Medium": 1, "High": 2}
        #ordinal
        self.df["Item_Fat_Content"] = self.df["Item_Fat_Content"].map(Item_Fat_Content_mapping)
        self.df["Outlet_Size"] = self.df["Outlet_Size"].map(Outlet_Size_mapping)
        #nominal
        self.df_ohe = pd.get_dummies(self.df,drop_first=True)
        
    # def scaling(self):
        for col in self.num_cols:
            min_val = self.df_ohe[col].min()
            max_val = self.df_ohe[col].max()
            self.df_ohe[col] = (self.df_ohe[col] - min_val)/(max_val - min_val)
        
        
    # def train_validation_split(self):
        X = self.df_ohe.drop("Item_Outlet_Sales", axis = 1)
        y = self.df_ohe["Item_Outlet_Sales"]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, random_state=42, test_size=0.3)
        
    
        # return pd.concat([self.df2["Item_Identifier"],self.df_ohe], axis = 1)
        if not self.test:
            #defining neurons in each layer
            self.input_neurons = self.X_train.shape[1]
            self.neurons_hidden_layer_1 = 100
            self.neurons_hidden_layer_2 = 40
            self.neurons_hidden_layer_3 = 10
            self.output_neurons = 1
            
            self.model = Sequential()
            self.model.add(InputLayer(input_shape = (self.input_neurons,)))
            self.model.add(Dense(units = self.neurons_hidden_layer_1, activation=relu))
            self.model.add(Dense(units = self.neurons_hidden_layer_2, activation=relu))
            self.model.add(Dense(units = self.neurons_hidden_layer_3, activation=relu))
            self.model.add(Dense(units=1, activation=linear))
            
            ### Compiling the model - Defining the loss fucntion and optimizer
            self.model.compile(loss=keras.losses.MeanAbsoluteError, optimizer="adam", metrics = ["msle"])
            
            self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs = 300)
            self.model_evaluation()
        else:
            self.model_evaluation()
    def model_evaluation(self):
        self.pred_y_val = self.model.predict(self.X_val)
        mae = mean_absolute_error(self.y_val, self.pred_y_val)
        
        
        predicted = pd.DataFrame(self.pred_y_val, columns=["Item_Outlet_Sales"])
        result  = pd.concat([self.df2["Item_Identifier"], self.df2["Outlet_Identifier"], predicted["Item_Outlet_Sales"]], axis = 1)
        result.to_csv("final_result.csv",index = False)
        return mae
    
if __name__ == "__main__":
    train = Preprocessing("train.csv")
    mae  = train.pre_processed_data()
    print("MAE:",mae)
    # df.to_csv("preprocessed_train_data.csv", index =False)
    test = Preprocessing("test.csv",test = True)
    mae2 = test.pre_processed_data()
    print("MAE:",mae2)