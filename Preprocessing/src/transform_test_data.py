from PCA_and_LDA import get_label_index
import os
import cv2
import numpy as np


def load_flattened_images(base_folder: str):
    image_list = []
    labels = []
    for person_id_folder in os.listdir(base_folder):
        person_folder_path = os.path.join(base_folder, person_id_folder)
        if not os.path.isdir(person_folder_path):
            continue
        for image_name in os.listdir(person_folder_path):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_folder_path, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    image_list.append(img.flatten())
                    class_label = os.path.splitext(image_name)[0]
                    label_index = get_label_index(class_label)
                    labels.append(label_index)
    X = np.array(image_list, dtype=np.float32) / 255.0
    return X, labels

def transform_test_data(test_folder: str, pca, lda, mean_image, u: int = 5):

    """
    Transformerer testdata ved hjelp av forhåndstrente PCA og LDA modeller.
    Args:
        test_folder (str): Stien til mappen som inneholder testbilder.
        pca: Forhåndstrent PCA-modell.
        lda: Forhåndstrent LDA-modell.
        mean_image (np.ndarray): Gjennomsnittsbildet brukt under trening.
        u (int): Antall første PCA-komponenter som skal kastes.
    Returns:
        Y_fisher_test (np.ndarray): Testdata etter LDA-transformasjon.
        Z_pca_test (np.ndarray): Testdata etter PCA-transformasjon.
        labels_test (list): Liste over merkelapper for testbildene.
    """
    X_test, labels_test = load_flattened_images(test_folder)
    # Sentrer testdata med mean_image fra treningsdata
    A_test_centered = X_test - mean_image
    Z_pca_test = pca.transform(A_test_centered)
    if u > 0:
        Z_pca_test = Z_pca_test[:, u:]
    Y_fisher_test = lda.transform(Z_pca_test)
    return Y_fisher_test,Z_pca_test, labels_test