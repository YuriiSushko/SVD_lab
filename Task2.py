import pandas as pd


file_path = 'ml-latest-small/ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=15, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=50, axis=1)

mean = ratings_matrix.mean()
min = ratings_matrix.min()
max = ratings_matrix.max()

ratings_matrix_Nan_mean = ratings_matrix.fillna(mean)
ratings_matrix_Nan_min = ratings_matrix.fillna(min)
ratings_matrix_Nan_max = ratings_matrix.fillna(max)
