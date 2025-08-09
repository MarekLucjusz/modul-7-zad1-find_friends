# app.py — Streamlit + PyCaret clustering (wersja gotowa na Streamlit Community Cloud)
# ---------------------------------------------------------------
# Najważniejsze cechy:
# - Czytelne ścieżki (Pathlib) i przewidywalna struktura katalogów: models/, data/, cfg/
# - Cache modelu: st.cache_resource (kosztowne wczytywanie wykonywane raz)
# - Cache danych i meta: st.cache_data (szybkie, z „podpisem” modelu, by aktualizować cache po zmianie .pkl)
# - Naprawa błędu UnhashableParamError: argument modelu w cache oznaczony jako _model (z podkreśleniem)
# - Odporność na brak plików i brak kolumn w danych
# - Wykresy Plotly z use_container_width

import json                                # do wczytywania nazw/opisów klastrów z pliku JSON
import os                                  # do statystyk plików (mtime/rozmiar) przy podpisie modelu
from pathlib import Path                   # bezpieczne ścieżki niezależne od systemu
import streamlit as st                     # framework webowy do budowy aplikacji
import pandas as pd                        # analiza danych tabelarycznych
from pycaret.clustering import load_model, predict_model  # wczytanie i predykcja modelu klastrującego
import plotly.express as px                # wykresy interaktywne

# -----------------------
# USTAWIENIA I ŚCIEŻKI
# -----------------------
st.set_page_config(page_title="Rekomendacje grupy (Clustering)", page_icon="🧩", layout="wide")  # konfiguracja strony

ROOT = Path(__file__).parent              # katalog, w którym znajduje się app.py
MODELS_DIR = ROOT / "models"              # katalog na modele (np. models/model_4.pkl)
DATA_DIR = ROOT / "data"                  # katalog na dane CSV (np. data/data.csv)
CFG_DIR = ROOT / "cfg"                    # katalog na pliki konfiguracyjne (np. cfg/nazwy_klastrow_4.json)

MODEL_NAME = "model_4"                    # nazwa bazowa modelu (bez rozszerzenia)
DATA_FILE = DATA_DIR / "data.csv"         # nazwa pliku z danymi; oczekiwany separator ';'
CLUSTERS_JSON = CFG_DIR / "nazwy_klastrow_4.json"  # nazwy i opisy klastrów w formacie JSON

# -----------------------
# FUNKCJE CACHE: MODEL
# -----------------------
@st.cache_resource(show_spinner=True)     # cache zasobów — idealne do wczytywania kosztownego modelu
def get_model(model_name: str):
    """
    Wczytuje model PyCaret z katalogu models/.
    PyCaret load_model przyjmuje ścieżkę bez rozszerzenia .pkl, więc usuwamy sufiks.
    """
    # Przykład: models/model_4 (bez .pkl) — PyCaret sam dopełni rozszerzenie podczas wczytywania
    model_path_no_ext = str((MODELS_DIR / model_name).with_suffix(""))
    return load_model(model_path_no_ext)

def get_model_signature(model_name: str) -> str:
    """
    Zwraca 'podpis' modelu (string), aby cache danych odświeżał się po podmianie pliku .pkl.
    Wykorzystujemy czas modyfikacji i rozmiar pliku. Jeśli pliku brak — fallback do samej nazwy.
    """
    try:
        pkl_path = (MODELS_DIR / model_name).with_suffix(".pkl")  # np. models/model_4.pkl
        stt = pkl_path.stat()                                     # pobierz statystyki pliku
        return f"{model_name}:{stt.st_mtime_ns}:{stt.st_size}"    # niezmienny, krótki hash kluczujący cache
    except Exception:
        return model_name                                         # awaryjnie sama nazwa

# --------------------------------------------
# FUNKCJE CACHE: META (JSON) + DANE UCZESTNIKÓW
# --------------------------------------------
@st.cache_data(show_spinner=True)          # szybkie, serializowalne dane — idealne do JSON i CSV
def get_cluster_names_and_descriptions(json_path: Path) -> dict:
    """
    Wczytuje słownik nazw i opisów klastrów z pliku JSON.
    Klucze w JSON zwykle są stringami ("0", "1", ...).
    """
    if not json_path.exists():
        st.error(f"Brak pliku z nazwami/objaśnieniami klastrów: {json_path}")
        st.stop()
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=True)
def get_all_participants(data_path: Path, _model, model_sig: str) -> pd.DataFrame:
    """
    Wczytuje dane uczestników i dokleja im przynależność do klastra.
    UWAGA 1: _model zaczyna się od '_' — Streamlit NIE próbuje go hashować (naprawa UnhashableParamError).
    UWAGA 2: model_sig to zwykły string (hash), więc cache odświeża się, gdy zmieni się plik .pkl.
    """
    if not data_path.exists():
        st.error(f"Brak pliku z danymi: {data_path}")
        st.stop()

    # Wczytaj dane źródłowe; jeśli masz polskie znaki — UTF-8 jest bezpieczne
    all_df = pd.read_csv(data_path, sep=';', encoding='utf-8')

    # Predykcja klastrów dla całego zbioru (predict_model zwraca DF z kolumną "Cluster")
    df_with_clusters = predict_model(_model, data=all_df)

    # Walidacja obecności kolumny "Cluster"
    if "Cluster" not in df_with_clusters.columns:
        st.error("Wynik predict_model nie zawiera kolumny 'Cluster'. Sprawdź zgodność modelu i danych.")
        st.stop()

    return df_with_clusters

# ------------------------------------------------
# BOCZNY PANEL (SIDEBAR): DANE OSOBY DO PREDYKCJI
# ------------------------------------------------
with st.sidebar:
    st.header("Powiedz nam coś o sobie")                              # nagłówek w sidebarze
    st.markdown("Pomożemy Ci znaleźć osoby o podobnych zainteresowaniach.")  # opis

    # Uwaga: wartości powinny odpowiadać tym, na których trenowano model (spójne kategorie!)
    age = st.selectbox("Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Koty i Psy', 'Inne'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    # Jednowierszowy DataFrame zgodny ze schematem treningowym modelu
    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
    }])

# ---------------------------
# WCZYTANIE ZASOBÓW + DANE
# ---------------------------
model = get_model(MODEL_NAME)                                   # wczytaj model (cache_resource => raz na sesję)
model_sig = get_model_signature(MODEL_NAME)                      # wyznacz podpis modelu (mtime/rozmiar pliku .pkl)
cluster_meta = get_cluster_names_and_descriptions(CLUSTERS_JSON) # wczytaj meta klastrów (JSON)
all_df = get_all_participants(DATA_FILE, model, model_sig)       # wczytaj dane + doklej klaster (cache_data)

# ---------------------------------
# PREDYKCJA KLASTRA DLA UŻYTKOWNIKA
# ---------------------------------
pred = predict_model(model, data=person_df)                      # predykcja klastra dla formularza z sidebaru
if "Cluster" not in pred.columns:                                # walidacja kolumny
    st.error("Brak kolumny 'Cluster' w wyniku predykcji. Sprawdź model i wejście.")
    st.stop()

predicted_cluster_id = pred["Cluster"].values[0]                 # np. 0, 1, 2, ...
predicted_cluster_id_str = str(predicted_cluster_id)             # klucze w JSON to zwykle stringi

# -----------------------
# NAZWA I OPIS KLASTRA
# -----------------------
if predicted_cluster_id_str not in cluster_meta:
    # Gdy w JSON nie ma wpisu dla danego klastra — zachowujemy się łagodnie
    cluster_name = f"Klaster {predicted_cluster_id}"
    cluster_desc = "Brak opisu dla tego klastra. Uzupełnij plik cfg/nazwy_klastrow_4.json."
else:
    # Pobieramy nazwę i opis z JSON (z domyślnymi wartościami na wypadek braków pól)
    cluster_name = cluster_meta[predicted_cluster_id_str].get("name", f"Klaster {predicted_cluster_id}")
    cluster_desc = cluster_meta[predicted_cluster_id_str].get("description", "Brak opisu dla tego klastra.")

# -----------------------
# PREZENTACJA WYNIKÓW
# -----------------------
st.header(f"Najbliżej Ci do grupy: {cluster_name}")              # wyświetlamy nazwę grupy
st.markdown(cluster_desc)                                        # wyświetlamy opis grupy

# Filtrujemy osoby z takim samym klastrem (uważamy na typ porównania: int vs str)
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]

# KPI: liczba osób w wybranym klastrze
st.metric("Liczba osób w Twojej grupie", len(same_cluster_df))

# Jeśli brak danych — kończymy uprzejmie
if same_cluster_df.empty:
    st.info("Brak osób w tej grupie (lub brak danych). Dodaj rekordy do data/data.csv.")
    st.stop()

# -----------------------
# WYKRESY (PLOTLY)
# -----------------------
# Uwaga: sprawdzamy istnienie kolumn, by nie wybuchać na brakującym polu

if "age" in same_cluster_df.columns:
    # Rozkład wieku — sortowanie po age poprawia czytelność histogramu kategorycznego
    fig = px.histogram(same_cluster_df.sort_values("age"), x="age", title="Rozkład wieku w grupie")
    fig.update_layout(xaxis_title="Wiek", yaxis_title="Liczba osób")
    st.plotly_chart(fig, use_container_width=True)  # dopasowanie do szerokości layoutu

if "edu_level" in same_cluster_df.columns:
    fig = px.histogram(same_cluster_df, x="edu_level", title="Rozkład wykształcenia w grupie")
    fig.update_layout(xaxis_title="Wykształcenie", yaxis_title="Liczba osób")
    st.plotly_chart(fig, use_container_width=True)

if "fav_animals" in same_cluster_df.columns:
    fig = px.histogram(same_cluster_df, x="fav_animals", title="Rozkład ulubionych zwierząt w grupie")
    fig.update_layout(xaxis_title="Ulubione zwierzęta", yaxis_title="Liczba osób")
    st.plotly_chart(fig, use_container_width=True)

if "fav_place" in same_cluster_df.columns:
    fig = px.histogram(same_cluster_df, x="fav_place", title="Rozkład ulubionych miejsc w grupie")
    fig.update_layout(xaxis_title="Ulubione miejsce", yaxis_title="Liczba osób")
    st.plotly_chart(fig, use_container_width=True)

if "gender" in same_cluster_df.columns:
    fig = px.histogram(same_cluster_df, x="gender", title="Rozkład płci w grupie")
    fig.update_layout(xaxis_title="Płeć", yaxis_title="Liczba osób")
    st.plotly_chart(fig, use_container_width=True)

# --- KONIEC PLIKU ---
