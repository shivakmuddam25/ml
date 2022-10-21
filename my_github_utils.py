def ROC_Curve(fpr, tpr, roc_auc):
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    # plt.plot(pred_prob1[1])
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
 
def import_basic_libraries():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly as ply
    import missingno
    import os
    from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    import sklearn

def find_null_values(df_eq_2):
    ls_nas = [df_eq_2.isna().sum() > 0]
    ls_nas

    if df_eq_2.isna().sum().sum() == 0:
        print("No Null values found in the DateFrame")
    elif df_eq_2.isna().sum().sum() > 0:
        null_cols = []
        null_cols_total = []
        null_cols_bool = []
        df2_null_per = []

        # Find null cols
        for col_num in range(len(df_eq_2.columns)):
            if ls_nas[0][col_num] == True:
        #         null_cols_bool.append(ls_nas[0][col_num])
                null_cols.append(df_eq_2.columns[col_num])
        #         df2_null_per.append(np.round(df_eq_2.columns[col_num].isna().sum()/df_eq_2.shape[0]*100,2))

        #         print(df_eq_2.columns[col_num], " : " + str(ls_nas[0][col_num]))


        for null_col in null_cols:
            df2_null_per.append(np.round(df_eq_2[null_col].isna().sum()/df_eq_2.shape[0]*100 , 2))
            null_cols_total.append(df_eq_2[null_col].isna().sum())
        #     print(df_eq_2[null_col].isna().sum())

        # d = pd.DataFrame(list(zip(null_cols, null_cols_bool)), columns= ["Df_2_Null_Cols", "null_cols_bool"])
        null_cols_per_df2 = pd.DataFrame(list(zip(null_cols, null_cols_total, df2_null_per)),
                         columns= ["DF Null Cols", "Total Null Values", "Percentage %"]).sort_values(by = ['Percentage %'], ascending = False)
        return null_cols_per_df2
