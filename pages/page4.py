from navigation import make_sidebar
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

make_sidebar()

# Fungsi untuk memuat dan memproses dataset
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    X = df.drop(['status', 'Nama'], axis=1)  # Menghapus kolom 'status' dan 'Nama'
    y = df['status']
    return df, X, y  # Mengembalikan DataFrame lengkap dan X, y

# Fungsi untuk melatih model Decision Tree dan menghitung akurasi
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()  # Menggunakan Decision Tree
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

# Fungsi untuk melakukan prediksi dengan model yang telah dilatih
def predict(model, X):
    return model.predict(X)

# Membuat antarmuka pengguna dengan Streamlit
st.title('Klasifikasi Kecemasan dengan Decision Tree')

uploaded_file = st.file_uploader("Upload dataset baru untuk klasifikasi:", type=['csv'])

if uploaded_file is not None:
    df_new, X_new, y_new = load_data(uploaded_file)  # Load dataset yang diunggah oleh pengguna
    st.write("Data yang akan diklasifikasikan:")
    st.write(df_new)

    # Load dataset kecemasan yang sudah ada
    df = pd.read_csv("kecemasan.csv")
    X = df.drop(['status', 'Nama'], axis=1) 
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    st.write(f"Akurasi Model: {accuracy*100:.2f}%")
    
    if st.button('Lakukan Klasifikasi'):
        predictions = predict(model, X_new)
        df_new['Prediksi Status'] = predictions
        df_new['Status Sebenarnya'] = y_new
        st.write("Hasil Klasifikasi:")
        st.write(df_new)

        # Tampilkan hasil prediksi ke dalam diagram batang
        st.header("Visualisasi Hasil Prediksi")
        
        # Diagram Batang
        st.subheader("Diagram Batang")
        pred_counts = df_new['Prediksi Status'].value_counts()
        plt.bar(pred_counts.index, pred_counts.values)
        plt.xlabel('Prediksi Status')
        plt.ylabel('Jumlah')
        st.pyplot()

        # Diagram Garis
        st.subheader("Diagram Garis")
        plt.plot(pred_counts.index, pred_counts.values, marker='o')
        plt.xlabel('Prediksi Status')
        plt.ylabel('Jumlah')
        st.pyplot()

        # Scatter Plot
        st.subheader("Scatter Plot")
        # Misalnya, asumsikan X_new memiliki 2 fitur
        label_to_color = {'TINGGI': 'red', 'SEDANG': 'blue', 'RENDAH': 'green'}
        colors = [label_to_color[label] for label in predictions]
        plt.scatter(X_new.iloc[:, 0], X_new.iloc[:, 1], c=colors, cmap='viridis')
        plt.xlabel('Fitur 1')
        plt.ylabel('Fitur 2')
        st.pyplot()


        # Tampilkan hasil prediksi ke dalam diagram lingkaran
        st.subheader("Diagram Lingkaran")
        plt.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
        st.pyplot()
else:
    st.write("Silakan unggah dataset baru dalam format CSV untuk dilakukan klasifikasi.")
