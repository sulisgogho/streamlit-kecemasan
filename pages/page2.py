from navigation import make_sidebar
import streamlit as st
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

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
        st.subheader("Group by Status (All Data):")
        status_group_all = df['status'].value_counts()
        st.write(status_group_all)
        
        # Memilih kolom sebagai fitur (p1-p17) dan target (status)
        X = df.iloc[:, 1:-1]  # Fitur
        y = df['status']      # Target
        
        # Memisahkan dataset menjadi data training dan testing dengan proporsi tetap dari setiap status
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Menampilkan jumlah data dalam setiap kelompok berdasarkan status untuk data training
        st.subheader("Group by Status (Training Data):")
        status_group_train = y_train.value_counts()
        st.write(status_group_train)
        
        # Menampilkan jumlah data dalam setiap kelompok berdasarkan status untuk data testing
        st.subheader("Group by Status (Testing Data):")
        status_group_test = y_test.value_counts()
        st.write(status_group_test)
        
        # Menampilkan nilai akurasi
        k = st.slider("Pilih nilai k:", min_value=1, max_value=10, value=5, step=1)
        accuracy = knn_accuracy(X_train, X_test, y_train, y_test, k)
        st.subheader("Akurasi KNN:")
        st.write(f"**Nilai akurasi dengan k = {k} :  {accuracy:.2f}**")

        # Menampilkan akurasi nilai k1-k10 dalam bentuk diagram garis
        st.subheader("Akurasi untuk Nilai k1-k10:")
        plot_accuracy(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
