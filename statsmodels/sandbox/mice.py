class ImputedData:

    def __init__(self, data, values):
        self.data = data
        self.values = values

   def toDataFrame(self):
       df = pd.DataFrame(self.data)
       for k in self.values.keys():
           ix = self.values[k][0]
           v = self.values[k][1]
           df[k][ix] = v
       return df

    def toArray(self):
        ar = np.asarray(self.data.copy())
        for k in self.values.keys():
            ix = self.values[k][0]
            v = self.values[k][1]
            ar[ii,k] = v
        return ar