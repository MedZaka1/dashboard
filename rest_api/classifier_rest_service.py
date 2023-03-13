from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)

with open('logreg.pickle', 'rb') as f:
    logreg = pickle.load(f)
    
with open('decision_tree.pickle', 'rb') as f:
    decision_tree = pickle.load(f)
    
with open('random_forest.pickle', 'rb') as f:
    random_forest = pickle.load(f)
    
with open('knn.pickle', 'rb') as f:
    knn = pickle.load(f)
    
with open('svm.pickle', 'rb') as f:
    svm = pickle.load(f)
    
with open('standard_scaler.pickle', 'rb') as f:
    standard_scaler = pickle.load(f)
    
with open('minmax_scaler.pickle', 'rb') as f:
    minmax_scaler = pickle.load(f)
    
with open('normalizer.pickle', 'rb') as f:
    normalizer = pickle.load(f)

def predict_logreg(X):
    X_transformed = standard_scaler.transform(X)
    y_pred = logreg.predict(X_transformed)
    return y_pred

def predict_decision_tree(X):
    X_transformed = minmax_scaler.transform(X)
    y_pred = decision_tree.predict(X_transformed)
    return y_pred

def predict_random_forest(X):
    X_transformed = normalizer.transform(X)
    y_pred = random_forest.predict(X_transformed)
    return y_pred

def predict_knn(X):
    X_transformed = standard_scaler.transform(X)
    y_pred = knn.predict(X_transformed)
    return y_pred

def predict_svm(X):
    X_transformed = minmax_scaler.transform(X)
    y_pred = svm.predict(X_transformed)
    return y_pred

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    X = data['X']
    model_name = data['model']
    transformation_name = data['transformation']
    
    if model_name == 'logreg':
        model = predict_logreg
    elif model_name == 'decision_tree':
        model = predict_decision_tree
    elif model_name == 'random_forest':
        model = predict_random_forest
    elif model_name == 'knn':
        model = predict_knn
    elif model_name == 'svm':
        model = predict_svm
    else:
        return jsonify({'error': 'Invalid model name'})
    
    if transformation_name == 'standard':
        transformer = StandardScaler()
    elif transformation_name == 'minmax':
        transformer = MinMaxScaler()
    elif transformation_name == 'pca':
        transformer = PCA(n_components=2)
    else:
        return jsonify({'error': 'Invalid transformation name'})
    X_transformed = transformer.transform(X)
    y_pred = model.predict(X_transformed)