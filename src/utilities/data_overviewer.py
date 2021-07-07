import pandas as pd


def data_characterization(data):
    """
    Give statistical insight of the data in a dataframe where the index is the column name.
    :param data: -containing statistical information for each column of the dataset.

    :return: dataframe containing statistical information for each column of the dataset
    """
    df = pd.DataFrame()
    Count = []
    final_value_count = []
    Nan_counts = data.isnull().sum().tolist()
    missing_vals = data.isnull().sum()
    Nan_ratio = []
    missing_val_percent = 100 * missing_vals / data.shape[0]
    columns = data.columns

    data_description = data.describe()
    statistical_info = data_description.loc[data_description.index[1:]].T
    statistical_info.reset_index(inplace = True)
    statistical_info.rename(columns={'index':'Columns_name'},inplace=True)
    
    #Attribute for each column the % of missing values
    for  col  in columns :
        for index,val in zip(missing_val_percent.index,missing_val_percent):
                if index==col :
                    Nan_ratio.append(val)
                    continue
        #Value count
        i = 0
        value_count = data[col].value_counts()
        value_counts = []
        #Store top 5 values for each column based on their occurence
        for val, occurence in zip(value_count.index,value_count.values):
            if i<=5:
                value_counts.append(str(val)+":"+str(occurence))
                i += 1
            else:
                break
        value_counts_String=""
        for val in value_counts:
            value_counts_String = value_counts_String + val + " "
        final_value_count.append(value_counts_String)
        Count.append(len(list(data[col].unique())))
    df["Columns_name"] = columns
    # df["Description"] = [self.dic.get(i)[1].get("description") for i in columns]
    df["Type"] = data.dtypes.tolist()
    df["Nb_unique_values"] = Count
    df["Nb_Nan_values"] = Nan_counts
    df["%_Nan_values"] = Nan_ratio
    df["Unique_values(value:count)"] = final_value_count
    df = df.merge(statistical_info,on="Columns_name",how="left")
    df.fillna("-",inplace=True)
    return df
