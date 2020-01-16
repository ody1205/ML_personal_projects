import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("iris.csv", encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기 
y = iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
y.replace({'Iris-setosa': 1, 'Iris-versicolor':2, 'Iris-virginica':3}, inplace = True)

# 학습 전용과 테스트 전용 분리하기 
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

# classifier 알고리즘 모두 추출하기--- (*1)
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

kfold_cv = KFold(n_splits=5, shuffle=True)

for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기 --- (*2)
    if name == 'ClassifierChain' or name == 'MultiOutputClassifier' or name == 'neVsOneClassifier' or name == 'OneVsOneClassifier' or name == 'OneVsRestClassifier' or name == 'OutputCodeClassifier' or name == 'StackingClassifier' or name == 'VotingClassifier':
        continue
    else:
        clf = algorithm()
        # 학습하고 평가하기 --- (*3)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(name,"의 정답률 = " , accuracy_score(y_test, y_pred))
        
    if hasattr(clf,'score'):
        scores = cross_val_score(clf, x, y, cv=kfold_cv)
        print(name,'의 정답률 = ',  scores)
