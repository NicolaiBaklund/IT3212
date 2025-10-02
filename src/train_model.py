
import os
import shutil
import random
from PCA_and_LDA import run_PCA_LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from transform_test_data import transform_test_data

def split_train_test(input_dir, train_dir, test_dir, train_ratio=0.8, seed=42):
    persons = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    random.seed(seed)
    random.shuffle(persons)
    split_idx = int(len(persons) * train_ratio)
    train_persons = persons[:split_idx]
    test_persons = persons[split_idx:]
    for person in train_persons:
        shutil.copytree(os.path.join(input_dir, person), os.path.join(train_dir, person), dirs_exist_ok=True)
    for person in test_persons:
        shutil.copytree(os.path.join(input_dir, person), os.path.join(test_dir, person), dirs_exist_ok=True)
    return train_persons, test_persons

def cross_validate_pca_lda(input_dir, k=50, u=5, num_folds=5):
    results = []
    results_pca = []
    # Antall unike klasser i datasettet
    num_classes = 8  #Juster dette basert på datasettet
    for fold in range(num_folds):
        train_dir = os.path.join(input_dir, f"train_fold_{fold}")
        test_dir = os.path.join(input_dir, f"test_fold_{fold}")
        # Rydd opp gamle fold-mapper hvis de finnes
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        # Split data
        train_persons, test_persons = split_train_test(input_dir, train_dir, test_dir, seed=fold)
        # Kjør PCA+LDA på treningsdata
        Y_train,Z_PCA_train, pca_train, lda_train, mean_image_train, labels_train = run_PCA_LDA(train_dir, k=k, u=u)
        # Tren SVM på treningsdata
        svm = SVC(kernel='linear', random_state=42)
        svm.fit(Y_train, labels_train)
        
        # Forbered testdata
        Y_test,Z_PCA_test, labels_test = transform_test_data(test_dir, pca_train, lda_train, mean_image_train, u=u)
        
        
        # Evaluer på testdata
        preds = svm.predict(Y_test)
        acc = accuracy_score(labels_test, preds)
        results.append(acc)
        print(f"Fold {fold+1} test accuracy: {acc:.4f}")
        
        #Test av PCA uten LDA__________________________________________________________________________________________
        
        #tren  SVM på PCA data
        svm_pca = SVC(kernel='linear', random_state=42)
        svm_pca.fit(Z_PCA_train, labels_train)
        #Evaluer på PCA test data
        preds_pca = svm_pca.predict(Z_PCA_test)
        acc_pca = accuracy_score(labels_test, preds_pca)
        print(f"Fold {fold+1} test accuracy with PCA only: {acc_pca:.4f}")
        results_pca.append(acc_pca)
        
        # Slett fold-mappene for å spare plass
        shutil.rmtree(train_dir)
        shutil.rmtree(test_dir)
    #Vis gjennomsnitt og std av resultatene
    print(f"\nGjennomsnittlig nøyaktighet over {num_folds} fold med PCA+LDA: {sum(results)/num_folds:.4f} ± { (sum((x - sum(results)/num_folds) ** 2 for x in results) / num_folds) ** 0.5:.4f}")
    print(f"Gjennomsnittlig nøyaktighet over {num_folds} fold med PCA only: {sum(results_pca)/num_folds:.4f} ± { (sum((x - sum(results_pca)/num_folds) ** 2 for x in results_pca) / num_folds) ** 0.5:.4f}")
    
    return results
