import os
import cv2
import numpy as np
# pip install scikit-learn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


def run_PCA_LDA(base_folder: str, k: int = 200, u: int = 5, num_classes: int = 8, run_lda: bool = True):
    """
    Utfører PCA på bilder i base_folder.

    Args:
        base_folder (str): Sti til mappen med forbehandlede bilder.
        k (int): Antall PCA-komponenter som skal beholdes etter reduksjon.
        u (int): Antall første PCA-komponenter som skal kastes bort (f.eks. for å fjerne globale variasjoner).

    Returns:
        Z_pca (np.ndarray): Datamatrisen etter PCA og fjerning av de første u komponentene.
        pca (PCA): Trenet PCA-objekt fra scikit-learn.
        mean_image (np.ndarray): Gjennomsnittsbildet.
        labels (list): Tallverdier for klasselabels til hvert bilde.
    """
    A, mean_image, labels = load_and_prepare_data_for_pca(base_folder)
    #Tren PCA modellen
    pca = PCA(n_components=k+u)
    #Transformerte data
    Z_pca_training = pca.fit_transform(A)
    # Kast bort de første u komponentene
    if u > 0:
        Z_pca_training = Z_pca_training[:, u:]
    if not run_lda:
        return None,Z_pca_training, pca, None, mean_image, labels
    # LDA del__________________________________________________________________________________________
    
    n_components_lda = num_classes - 1  #Juster dette basert på antall klasser i datasettet

    # 1. Initialiser LDA-modellen
    lda = LinearDiscriminantAnalysis(n_components=n_components_lda)

    # 2. Tren og transformer dataene
    # LDA bruker både funksjonene (Z) og klassene (y) under treningen.
    Y_fisher = lda.fit_transform(Z_pca_training, labels)

    return Y_fisher,Z_pca_training, pca, lda, mean_image, labels


# Funksjon for å laste og forberede data for PCA_____________________________________________________________________________________
def load_and_prepare_data_for_pca(base_folder: str):
    """
    Laster inn alle forbehandlede bilder fra en mappestruktur på formen
    'base_folder/person_id/emotion.png', gjør dem om til en datamatrise X,
    og sentrerer dataene ved å trekke fra gjennomsnittsbildet.

    Args:
        base_folder: Stien til hovedmappen som inneholder person-undermapper.

    Returns:
        A (np.ndarray): Den sentrerte datamatrisen (X - X_bar).
        mean_image (np.ndarray): Det beregnede gjennomsnittsbildet.
        labels (list): En liste med merkelapper (labels) for hvert bilde.
    """
    image_list = []
    labels = []
    
    print(f"Laster inn bilder fra: {base_folder}")

    # Steg 1: Gå gjennom hver person-mappe
    for person_id_folder in os.listdir(base_folder):
        person_folder_path = os.path.join(base_folder, person_id_folder)
        if not os.path.isdir(person_folder_path):
            continue
            
        # Gå gjennom hver bildefil (f.eks. 'fear.png') i person-mappen
        for image_name in os.listdir(person_folder_path):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_folder_path, image_name)
                
                # Les bildet i gråtoner
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Gjør bildet om til en flat 1D-vektor
                    image_list.append(img.flatten())
                    
                    # Hent ut klassen fra filnavnet (fjerner f.eks. '.png')
                    class_label = os.path.splitext(image_name)[0]
                    label_index = get_label_index(class_label)
                    labels.append(label_index)

    # Konverter listen med bilder til en NumPy-matrise X
    X = np.array(image_list, dtype=np.float32)

    if X.shape[0] == 0:
        raise ValueError("Ingen bilder ble funnet. Sjekk stien og mappestrukturen.")

    print(f"Fullført. Lastet inn {X.shape[0]} bilder.")
    print(f"Dimensjoner på datamatrisen X: {X.shape}")

    # Normaliser dataene til området 0.0 - 1.0
    X = X / 255.0

    # Steg 2: Beregn gjennomsnittsbildet (X_bar)
    mean_image = np.mean(X, axis=0)
    print(f"Dimensjoner på gjennomsnittsbildet: {mean_image.shape}")

    # Steg 3: Sentrering (Mean Subtraction) for å få matrisen A
    A = X - mean_image
    print("Data er sentrert ved å trekke fra gjennomsnittsbildet.")
    
    return A, mean_image, labels



#Helperfunksjon__________________________________________________________________________________________

# Definerer de mulige klassene og deres indekser
class_names = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

def get_label_index(class_label):
    for idx, name in enumerate(class_names):
        if name in class_label.lower():
            return idx
    raise ValueError(f"Ukjent klasselabel i filnavn: {class_label}")

def visualize_fisherfaces(base_folder, k=70, u=0, num_classes=8, num_fisherfaces=7):
    Y_fisher, Z_pca, pca, lda, mean_image, labels = run_PCA_LDA(base_folder, k=k, u=u, num_classes=num_classes, run_lda=True)
    eigenfaces = pca.components_[u:]
    fisherfaces = lda.scalings_.T  # evt. lda.coef_
    img_size = int(np.sqrt(eigenfaces.shape[1]))

    print("eigenfaces shape:", eigenfaces.shape)
    print("fisherfaces shape:", fisherfaces.shape)
    print("Eksempel på fisherface-vektor:", fisherfaces[0])
    print("Unike verdier i fisherfaces:", np.unique(fisherfaces))

    # Visualiser fisherface-vektorene direkte
    plt.figure()
    for i in range(min(num_fisherfaces, fisherfaces.shape[0])):
        plt.subplot(1, num_fisherfaces, i+1)
        plt.plot(fisherfaces[i])
        plt.title(f'Fisherface vector {i+1}')
    plt.show()

    # Visualiser eigenfaces direkte
    plt.figure()
    for i in range(min(7, eigenfaces.shape[0])):
        plt.subplot(1, 7, i+1)
        plt.imshow(eigenfaces[i].reshape((img_size, img_size)), cmap='gray')
        plt.title(f'Eigenface {i+1}')
        plt.axis('off')
    plt.show()

    # Visualiser Fisherfaces som bilder
    plt.figure(figsize=(14, 2))
    for i in range(min(num_fisherfaces, fisherfaces.shape[0])):
        fisherface_pca = fisherfaces[i]
        fisherface_img = np.dot(fisherface_pca, eigenfaces)
        fisherface_img = fisherface_img + mean_image
        # Prøv uten normalisering først
        plt.subplot(1, num_fisherfaces, i+1)
        plt.imshow(fisherface_img.reshape((img_size, img_size)), cmap='gray')
        plt.title(f'Fisherface {i+1}')
        plt.axis('off')
    plt.suptitle("De første Fisherfacene")
    plt.show()