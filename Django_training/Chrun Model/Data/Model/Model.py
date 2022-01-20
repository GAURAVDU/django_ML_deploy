import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

data = pd.read_csv(r'C:\Users\imgau\Documents\Python Code Directory\Django_training\Chrun Model\Data\archive\glass.csv')
#print(data.head())

print('There are ',data.shape[0] ,'rows and ',data.shape[1],' columns.')

y= data['Type']
x= data.drop(['Type'], axis=1)

from sklearn.model_selection import train_test_split, cross_val_score
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV,KFold

def result_using_tree_classifier(x_train,x_test,y_train,y_test):
    kf = KFold(n_splits=5,random_state=None)
    scores= []
    algos2={'decision tree': {'model' : DecisionTreeClassifier(random_state=42),
                      'param' : {'criterion':['gini', 'entropy'],'max_depth' : np.arange(1,6,1).tolist()  }
                     },
           'random forest': {'model' : RandomForestClassifier(random_state=42, n_jobs= -1),
                      'param' : {'n_estimators': [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]}
                     },
           'extra tree' : {'model' : ExtraTreesClassifier(random_state=42, n_jobs= -1),
                    'param' : {'n_estimators': [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]}
                     } 
           } 
    for algo_name, params in algos2.items() :
        Grid2 = GridSea
        rchCV( params['model'], params['param'], cv=kf, return_train_score=False)
        Grid2.fit(x_train, y_train)
        ypred= Grid2.predict(x_test)
        import joblib
        file_name = algo_name+".sav"
        joblib.dump(Grid2,file_name)
        scores.append({
            'model' : algo_name,
            'best_score': Grid2.best_score_,
            'best_para': Grid2.best_params_
        })
    return pd.DataFrame(scores,columns=['model','best_score','best_para']).set_index('model').reset_index()


print(result_using_tree_classifier(x_train,x_test,y_train,y_test))

