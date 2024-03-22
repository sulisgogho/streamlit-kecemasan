from navigation import make_sidebar
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

make_sidebar()

# Fungsi untuk menghitung akurasi menggunakan algoritma KNN
def knn_accuracy(X_train, X_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Fungsi untuk menampilkan akurasi nilai k1-k10 dalam bentuk diagram garis
def plot_accuracy(X_train, X_test, y_train, y_test):
    k_values = range(1, 11)
    accuracies = []
    for k in k_values:
        accuracy = knn_accuracy(X_train, X_test, y_train, y_test, k)
        accuracies.append(accuracy)

    plt.plot(k_values, accuracies, marker='o')
    plt.title('Akurasi KNN untuk Nilai k')
    plt.xlabel('Nilai k')
    plt.ylabel('Akurasi')
    
    # Batasi sumbu y antara 0.95 dan 1.0
    plt.ylim(0.95, 1.0)
    
    # Atur nilai xticks (nilai k) secara eksplisit
    plt.xticks(k_values)
    
    # Convert Matplotlib plot to Streamlit
    st.pyplot()

# Fungsi untuk plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
# Fungsi untuk menampilkan confusion matrix
def plot_confusion_matrix_display(X_train, X_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Menampilkan confusion matrix
    st.subheader("Confusion Matrix:")
    st.write(pd.DataFrame(cm, columns=knn.classes_, index=knn.classes_))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    # Tambahkan keterangan warna
    color_bar = plt.colorbar()
    color_bar.set_label('Number of samples')
    
    st.pyplot()

# Fungsi untuk menampilkan confusion matrix
def plot_confusion_matrix_display(X_train, X_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Menampilkan confusion matrix
    st.subheader("Confusion Matrix:")
    st.write(pd.DataFrame(cm, columns=knn.classes_, index=knn.classes_))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    # Tambahkan keterangan warna
    im = plt.imshow(cm, cmap='Blues')
    color_bar = plt.colorbar(im)
    color_bar.set_label('Number of samples')
    
    st.pyplot()

# Fungsi untuk menampilkan classification report
def show_classification_report(X_train, X_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # Menampilkan classification report
    st.subheader("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Main Streamlit app
def main():
    st.title("Aplikasi Klasifikasi Kecemasan dengan KNN")
    
    # Upload dataset
    uploaded_file = st.file_uploader("Upload dataset CSV", type="csv")
    
    if uploaded_file is not None:
        # Memuat dataset ke dalam DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Menampilkan dataset
        st.subheader("Dataset")
        st.write(df)
        
        # Handling missing values
        df = df.dropna()  # Drop rows with missing values
        
        # Menampilkan group by status untuk seluruh dataset
        st.subheader("Group by Diagnosa (All Data):")
        status_group_all = df['Diagnosa'].value_counts()
        st.write(status_group_all)
        
        # Memilih kolom sebagai fitur (p1-p17) dan target (status)
        X = df.iloc[:, 1:-1]  # Fitur
        y = df['Diagnosa']      # Target
        
        # Memisahkan dataset menjadi data training dan testing dengan proporsi tetap dari setiap status
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Menampilkan jumlah data dalam setiap kelompok berdasarkan status untuk data training
        st.subheader("Group by Diagnosa (Training Data):")
        status_group_train = y_train.value_counts()
        st.write(status_group_train)
        
        # Menampilkan jumlah data dalam setiap kelompok berdasarkan status untuk data testing
        st.subheader("Group by Diagnosa (Testing Data):")
        status_group_test = y_test.value_counts()
        st.write(status_group_test)

        # Menampilkan akurasi nilai k1-k10 dalam bentuk diagram garis
        st.subheader("Akurasi untuk Nilai k1-k10:")
        plot_accuracy(X_train, X_test, y_train, y_test)
        
        # Menampilkan confusion matrix
        k = st.slider("Pilih nilai k:", min_value=1, max_value=10, value=5, step=1)
        plot_confusion_matrix_display(X_train, X_test, y_train, y_test, k)
        
        # Menampilkan classification report
        show_classification_report(X_train, X_test, y_train, y_test, k)
        
        # Menampilkan nilai akurasi
        accuracy = knn_accuracy(X_train, X_test, y_train, y_test, k)
        st.subheader("Akurasi KNN:")
        st.write(f"**Nilai akurasi dengan k = {k} :  {accuracy:.2f}**")

if __name__ == "__main__":
    main()
