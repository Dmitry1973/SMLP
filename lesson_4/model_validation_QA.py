import time
from datetime import datetime, timedelta
import pandas as pd
from WOE_IV import data_vars
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

def prepare_dataset(dataset,
                    dataset_type='train',
                    dataset_path='dataset/'):
    print(dataset_type)
    start_t = time.time()
    print('Dealing with missing values, outliers, categorical features...')

    # Профили
    dataset['age'] = dataset['age'].fillna(dataset['age'].median())
    dataset['gender'] = dataset['gender'].fillna(dataset['gender'].mode()[0])
    dataset.loc[~dataset['gender'].isin(['M', 'F']), 'gender'] = dataset['gender'].mode()[0]
    dataset['gender'] = dataset['gender'].map({'M': 1., 'F': 0.})
    dataset.loc[(dataset['age'] > 80) | (dataset['age'] < 7), 'age'] = round(dataset['age'].median())
    dataset.loc[dataset['days_between_fl_df'] < -1, 'days_between_fl_df'] = -1
    # Пинги
    for period in range(1, len(INTER_LIST) + 1):
        col = 'avg_min_ping_{}'.format(period)
        dataset.loc[(dataset[col] < 0) |
                    (dataset[col].isnull()), col] = dataset.loc[dataset[col] >= 0][col].median()
    # Сессии и прочее
    dataset.fillna(0, inplace=True)
    dataset.to_csv('{}dataset_{}.csv'.format(dataset_path, dataset_type), sep=';', index=False)

    print('Dataset is successfully prepared and saved to {}, run time (dealing with bad values): {}'. \
          format(dataset_path, time_format(time.time() - start_t)))


prepare_dataset(dataset=train, dataset_type='train')
prepare_dataset(dataset=test, dataset_type='test')

train_new = pd.read_csv('dataset/dataset_train.csv', sep=';')
# test_new = pd.read_csv('dataset/dataset_test.csv', sep=';')


dataset_raw = pd.read_csv('dataset/dataset_raw_train.csv', sep=';')
X_raw = dataset_raw.drop(['user_id', 'is_churned'], axis=1)
y_raw = dataset_raw['is_churned']

iv_df, iv = data_vars(X_raw, y_raw)

IV = iv.sort_values('IV', ascending=False)
plt.figure(figsize=(16,6))
plt.bar(range(IV.shape[0]), IV['IV'], align='center')
plt.xticks(range(IV.shape[0]), IV['VAR_NAME'].values, rotation=90)
plt.title('Information Value')
plt.show()

STEP = 5
logit = LogisticRegression(random_state=42)

selector = RFECV(estimator=logit, step=STEP, cv=StratifiedKFold(2), scoring='f1')
selector.fit(X_train_balanced, y_train_balanced)

good_features = X.columns[selector.support_]
print("Optimal number of features : %d" % selector.n_features_)