from navigation import make_sidebar
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

make_sidebar()

# Fungsi untuk memuat dan memproses dataset
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    X = df.drop(['Diagnosa', 'Nama'], axis=1)  # Menghapus kolom 'status' dan 'Nama'
    y = df['Diagnosa']
    return df, X, y  # Mengembalikan DataFrame lengkap dan X, y

# Fungsi untuk melatih model KNN dan menghitung akurasi
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

# Fungsi untuk melakukan prediksi dengan model yang telah dilatih
def predict(model, X):
    return model.predict(X)

# Membuat antarmuka pengguna dengan Streamlit
st.title('Klasifikasi Kecemasan dengan KNN')

uploaded_file = st.file_uploader("Upload dataset baru untuk klasifikasi:", type=['csv'])

if uploaded_file is not None:
    df_new, X_new, _ = load_data(uploaded_file)  # Load dataset yang diunggah oleh pengguna
    st.write("Data yang akan diklasifikasikan:")
    st.write(df_new)

    # Load dataset kecemasan yang sudah ada
    df = pd.read_csv("kecemasanatlet.csv")
    X = df.drop(['Diagnosa', 'Nama'], axis=1)
    y = df['Diagnosa']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    X_new_imputed = imputer.transform(X_new)

    model, accuracy = train_and_evaluate(X_train_imputed, X_test_imputed, y_train, y_test)
    
    st.write(f"Akurasi Model: {accuracy*100:.2f}%")
    
    if st.button('Lakukan Klasifikasi'):
        predictions = predict(model, X_new_imputed)
        df_new['Klasifikasi Status'] = predictions
        st.write("Hasil Klasifikasi:")
        st.write(df_new)

        # Tampilkan hasil prediksi ke dalam diagram batang
        st.header("Visualisasi Hasil Klasifikasi")
        
        # Diagram Batangs
        st.subheader("Diagram Batang")
        pred_counts = df_new['Klasifikasi Status'].value_counts()
        plt.bar(pred_counts.index, pred_counts.values)
        plt.xlabel('Klasifikasi Status')
        plt.ylabel('Jumlah')
        st.pyplot()

        # Diagram Garis
        st.subheader("Diagram Garis")
        plt.plot(pred_counts.index, pred_counts.values, marker='o')
        plt.xlabel('Prediksi Status')
        plt.ylabel('Jumlah')
        st.pyplot()

        # Tampilkan hasil prediksi ke dalam diagram lingkaran
        st.subheader("Diagram Lingkaran")
        plt.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
        st.pyplot()
else:
    st.write("Silakan unggah dataset baru dalam format CSV untuk dilakukan klasifikasi.")
