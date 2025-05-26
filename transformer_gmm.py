import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from transformer import *

def load_ucihar_data(path='UCI_HAR_Dataset/'):
    def load_file(file_path):
        return pd.read_csv(file_path, delim_whitespace=True, header=None).values
    X_train = load_file(path + 'train/X_train.txt')
    y_train = load_file(path + 'train/y_train.txt').flatten() - 1
    X_test = load_file(path + 'test/X_test.txt')
    y_test = load_file(path + 'test/y_test.txt').flatten() - 1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long))


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_ucihar_data()
    model = HARTransformerClassifier(num_layers=12)
    classifier = nn.Linear(128, 6)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        features = model(X_train)
        logits = classifier(features)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        train_features = model(X_train).cpu().numpy()
        test_features = model(X_test).cpu().numpy()

    gmm = GaussianMixture(n_components=6, random_state=0)
    gmm.fit(train_features.astype(np.float32))
    cluster_ids = gmm.predict(train_features)

    label_map = {}
    for c in range(6):
        indices = (cluster_ids == c)
        if np.sum(indices) > 0:
            label = mode(y_train.numpy()[indices], keepdims=False).mode
            label_map[c] = int(label)

    pred_clusters = gmm.predict(test_features)
    mapped_preds = np.vectorize(label_map.get)(pred_clusters)
    acc = np.mean(mapped_preds == y_test.numpy())
    print(f"Accuracy: {acc*100:.2f}%")
