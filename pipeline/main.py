from datetime import datetime

import dill
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def filter_data(df):
    # Удаляем не нужные признаки
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return df.drop(columns_to_drop, axis=1)


def remove_outliers(df):
    # Удаляем выборсы с помощью нахождения интерквартильного размаха
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

        return boundaries
    df = df.copy()

    boundaries = calculate_outliers(df['year'])

    df.loc[df['year'] < boundaries[0], 'year'] = round(boundaries[0])
    df.loc[df['year'] > boundaries[1], 'year'] = round(boundaries[1])

    return df


def create_features(df):

    df = df.copy()

    def short_model(x):
        import pandas
        if not pandas.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x

    # Добавляем фичу "short_model" – это первое слово из колонки model
    df.loc[:, 'short_model'] = df['model'].apply(short_model)

    # Добавляем фичу "age_category" (категория возраста)
    df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))

    return df


def main():
    # Загружаем датафрейм и делим его на признаки и целевую переменную.
    # Объявляем трансормеры для признаков и составляем пайплайн.
    df = pd.read_csv('data/data for model.csv')

    X = df.drop('price_category', axis=1)
    y = df['price_category']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('outlier_remover', FunctionTransformer(remove_outliers)),
        ('feature_creator', FunctionTransformer(create_features)),
        ('column_transformer', column_transformer)
    ])

    models = [
        # Объявляем массив с моделями для дальнейшого выбора лучшего варианта
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    ]

    best_score = .0
    best_pipe = None
    # Объявляем пустые переменные для дальнейший записи в них лучшей метрики и лучшей модели
    for model in models:

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

    best_pipe.fit(X, y)
    with open('cars_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Car price prediction model',
                'author': 'Gleb Vorobevskiy',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()