import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
class titanic:
    def __init__(self,input):
        data = pd.read_excel(input)
        self._dataframe = data


    def cleandataset(self):
        self._dataframe.drop(['body','name'],1,inplace=True)
        self._dataframe.convert_objects(convert_numeric=True)
        self._dataframe.fillna(0,inplace = True)
        return self._dataframe
    def label_text_to_init(self):
        self.cleandata = self.cleandataset()
        encoding = preprocessing.LabelEncoder()
        columns = self.cleandata.columns.values
        for column in columns:
            if self.cleandata[column].dtype != np.int64 and self.cleandata[column].dtype != np.float64:

                encoding.fit(self.cleandata[column])

                encoding.classes_
                self.cleandata['label_'+column]= encoding.transform(self.cleandata[column])
        return self.cleandata

    def printdata(self):

        print self.cleandataset().head()

if __name__ == '__main__':
    input = '/Users/badrkhamis/Desktop/Python_Pandas/clustering/titanic.xls'
    opject1 = titanic(input)
    print opject1.label_text_to_init()
