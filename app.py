import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


file_path = 'train.csv'
data = pd.read_csv(file_path)


X = data.drop('TARGET(PRICE_IN_LACS)', axis=1)
y = data['TARGET(PRICE_IN_LACS)']


X = X.drop('ADDRESS', axis=1)


label_encoder = LabelEncoder()
X['BHK_OR_RK'] = label_encoder.fit_transform(X['BHK_OR_RK'])
X['POSTED_BY'] = label_encoder.fit_transform(X['POSTED_BY'])


X = pd.get_dummies(X, columns=['BHK_OR_RK', 'POSTED_BY'], drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBRegressor()


param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}


grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


import joblib
joblib.dump(best_model, 'xgboost_model.joblib')
