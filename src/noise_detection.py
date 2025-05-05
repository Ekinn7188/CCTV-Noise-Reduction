import polars as pl
from .hog import HOG
from glob import glob
import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import multilabel_confusion_matrix

def generate_first_std_features(data):
    dfs = []

    for key in data.keys():
        loop_df = pl.from_dicts(list(data[key]))

        loop_df = loop_df.with_columns(pl.lit(key).alias("noise_type"))
        loop_df = loop_df.drop("noise_type").insert_column(0, loop_df.get_column("noise_type"))

        dfs.append(loop_df)

    df = pl.concat(dfs)

    return df

def gather_noise_examples_helper(folder):
        folder_basename = os.path.basename(folder[:-1])

        results = [] 
        for file in tqdm.tqdm(glob(folder + "/*.jpg"), desc=folder_basename):
            hog = HOG(file)
            
            results.append([hog.features])

        results = np.array(results)
        
        results = [{i:x[i].item() for i in x.keys()} for x in results[:,0]]

        results = np.array(results)

        if folder_basename not in ["saltpepper", "speckle"]:
            folder_basename = "other"

        return (folder_basename, results)

def gather_noise_examples(generate_train=False):
    final_results = {}

    generated = False

    if not os.path.exists("data/hog_features.pkl") or generate_train:
        print("Gathering noise examples...")
        
        results = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(gather_noise_examples_helper)(folder) for folder in glob("data/noise/train_im/*/"))

        for folder, result in results:
            if folder in final_results:
                final_results[folder] = np.concatenate((final_results[folder], result), axis=0)
            else:   
                final_results[folder] = result
        pickle.dump(final_results, open("data/hog_features.pkl", "wb"))
        
        print("Done.")

        generated = True
    else:
        final_results = pickle.load(open("data/hog_features.pkl", "rb"))

    return generate_first_std_features(final_results), generated

def predict_noise_type(hog, df):
    features = np.array([*hog.features.values()], dtype=np.float64)

    train_y = df.select(pl.col("noise_type")).to_numpy().flatten()
    train_x = df.select(pl.exclude("noise_type")).to_numpy()

    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    features_scaled = scaler.transform([features])

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_x_scaled, train_y)

    dist, idx = knn.kneighbors(features_scaled)
    
    labels = train_y[idx].flatten()
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = np.max(counts)
    max_count_labels = unique_labels[counts == max_count]
    
    if len(max_count_labels) > 1:
        valid_indices = np.isin(labels, max_count_labels)
        filtered_distances = dist[0][valid_indices]
        min_dist = np.min(filtered_distances)
        min_dist_idx = np.where((dist[0] == min_dist) & valid_indices)[0][0]
        best_vote = labels[min_dist_idx]
    else:
        best_vote = max_count_labels[0]

    return best_vote

def visualize_train_data(df):
    labels = df.select(pl.col("noise_type")).to_numpy().flatten()

    features = df.select(pl.exclude("noise_type")).to_numpy()

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    unique_labels = np.unique(labels)

    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label, s=30, alpha=0.7)

    plt.title("PCA Projection of Training Data", fontsize=20)
    plt.xlabel("PCA Component 1", fontsize=16)
    plt.ylabel("PCA Component 2", fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PCA.png")

def test_folder(folder, df):
    predicted = []
    real = []
    folder_basename = os.path.basename(folder[:-1])

    files = glob(folder + "*.jpg")
    
    if folder_basename not in ["saltpepper", "speckle"]:
        folder_basename = "other"

    for i, file in tqdm.tqdm(enumerate(files), desc=folder_basename, total=len(files)):
        hog = HOG(file)
        
        best_vote = predict_noise_type(hog, df)

        predicted.extend([best_vote])
        real.extend([folder_basename])

    return predicted, real

def model_evaluation(df):
    res = [test_folder(folder, df) for folder in glob("data/noise/test_im/*/")]

    predicted_votes = []
    real_votes = []
    for result in res:
        predicted, real = result
        if len(predicted) > 0:
            predicted_votes.extend(np.array(predicted).flatten().tolist())
            real_votes.extend(np.array(real).flatten().flatten().tolist())

    del res

    matrix = multilabel_confusion_matrix(real_votes, predicted_votes, labels=["saltpepper", "speckle", "other"])

    TP = matrix[:, 1, 1].sum()
    TN = matrix[:, 0, 0].sum()
    FP = matrix[:, 0, 1].sum()
    FN = matrix[:, 1, 0].sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return (matrix, TP, TN, FP, FN, accuracy, precision, recall, f1)