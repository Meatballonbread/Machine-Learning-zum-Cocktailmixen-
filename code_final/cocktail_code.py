"""
Dieses Programm erstellt basierend auf den Geschmackspräferenzen des Nutzers einen Cocktailvorschlag. 
Zunächst lädt es Daten zu Zutaten (mit ihren Geschmackseigenschaften wie süß, sauer, bitter, fruchtig, würzig) 
und Fragen aus einer JSON-Datei. Der Nutzer beantwortet Fragen, um sein persönliches Geschmacksprofil 
zu bestimmen. Anschließend kann er auswählen, ob der Cocktail alkoholisch oder nicht-alkoholisch sein soll. 

Mit Hilfe eines sogenannten Autoencoders (einer Form von Machine Learning) wird das Geschmacksprofil 
rekonstruiert, und ein mathematisches Verfahren (Least Squares) berechnet die optimale Mischung aus 
den passenden Zutaten. Dabei wird auch berücksichtigt, wie viele Zutaten der Cocktail maximal enthalten soll. 
Das Ergebnis ist ein individueller Cocktailvorschlag mit Mengenangaben für die Zutaten.

Die Mischung werden umso genauer, umso mehr Daten sich in der JSON-Datei befinden. 
Die Daten können beliebig erweitert werden, um die Auswahl an Zutaten und Geschmacksprofilen zu vergrößern.

Die Implementierung von Supervised Learning ist noch in Überlegung 

Author: Florian, Conrad
Date: 29.01.2025
"""

import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from datetime import datetime


def load_cocktail_data():
    """
    Lädt die Cocktail-Daten (Zutaten + Fragen) aus der JSON-Datei
    'code_final/cocktail_data_quest.json' (ggf. Pfad anpassen).
    """
    json_path = "code_final/cocktail_data_quest.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_and_train_autoencoder(ingredient_profiles, latent_dim=3, epochs=500, batch_size=4):
    """
    Erstellt einen einfachen Autoencoder und trainiert ihn auf den
    übergebenen Profilen (np.array shape=(n_samples, n_features)).
    """
    input_dim = ingredient_profiles.shape[1]
    
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(latent_dim, activation='relu')(input_layer)
    
    # Decoder
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    # Autoencoder
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Training
    autoencoder.fit(
        ingredient_profiles,
        ingredient_profiles,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0  # Falls du Trainingsausgaben sehen willst, setze auf 1 oder 2
    )
    
    return autoencoder


def create_mix_profile(reconstructed_profile, ingredient_profiles, ingredient_names, k, total_volume=200.0):
    """
    1) Least Squares für die Mischung aller Zutaten:
         min || A w - reconstructed_profile ||^2
       A = ingredient_profiles^T
    2) Negative Gewichte => 0
    3) Normalisieren => sum(w) = 1
    4) Nur Top k Gewichte behalten => Rest 0 => Normalisieren
    5) Gewichte * total_volume => ml

    Rückgabe: dict {Zutat: ml}
    """
    A = ingredient_profiles.T  # shape (features, M)
    b = reconstructed_profile
    
    # unbeschränktes least-squares
    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Negative auf 0 setzen
    w = np.maximum(w, 0)
    
    # Normalisieren => sum(w) = 1 (falls möglich)
    sum_w = np.sum(w)
    if sum_w == 0:
        # Fallback => Gleichverteilung
        w = np.ones_like(w) / len(w)
    else:
        w /= sum_w
    
    # Nur Top k behalten
    if k < len(w):
        sorted_idx = np.argsort(-w)  # absteigend sortiert
        top_k_idx = sorted_idx[:k]
        mask = np.zeros_like(w, dtype=bool)
        mask[top_k_idx] = True
        w[~mask] = 0
        # nochmal normalisieren
        sum_w_new = w.sum()
        if sum_w_new > 0:
            w /= sum_w_new
        else:
            w[mask] = 1.0 / k

    # ml anstatt (dimensionsloser) Gewichte
    volumes = w * total_volume
    
    mixture = {}
    for name, vol in zip(ingredient_names, volumes):
        if vol > 1e-6:
            mixture[name] = round(float(vol), 2)
    return mixture


def save_rating(user_profile, cocktail_mix, rating):
    """
    Speichert eine abgegebene Bewertung in einer JSON-Datei (z.B. 'cocktail_ratings.json').

    Wenn die Datei nicht existiert, wird sie neu erstellt und eine leere Struktur
    angelegt. Anschließend wird der neue Eintrag appended.
    """
    ratings_file = "code_final/cocktail_ratings.json"
    
    # Falls Datei nicht existiert, Grundstruktur anlegen
    if not os.path.exists(ratings_file):
        with open(ratings_file, "w", encoding="utf-8") as f:
            json.dump({"ratings": []}, f, ensure_ascii=False, indent=2)
    
    # Bestehende Daten laden
    with open(ratings_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Neuen Eintrag erstellen
    new_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_profile": list(map(float, user_profile)),  # in float konvertieren (falls np.float)
        "cocktail_mix": cocktail_mix,
        "rating": rating
    }
    
    # An die Liste anhängen
    data["ratings"].append(new_entry)
    
    # Zurückschreiben in die Datei
    with open(ratings_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("Deine Bewertung wurde erfolgreich gespeichert!")

def show_ratings():
    """
    Liest die bereits gespeicherten Bewertungen aus der JSON-Datei und zeigt
    sie in der Konsole an. Falls keine Bewertungen existieren, wird eine
    entsprechende Meldung ausgegeben.
    """
    ratings_file = "code_final/cocktail_ratings.json"
    
    if not os.path.exists(ratings_file):
        print("Keine Bewertungen vorhanden (Datei nicht gefunden).")
        return
    
    with open(ratings_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "ratings" not in data or len(data["ratings"]) == 0:
        print("Keine Bewertungen vorhanden.")
        return
    
    print("\n--- Vorhandene Bewertungen ---")
    for i, entry in enumerate(data["ratings"], start=1):
        print(f"Bewertung Nr. {i}:")
        print(f"  - Zeitpunkt:    {entry['timestamp']}")
        print(f"  - User-Profil:  {entry['user_profile']}")
        print(f"  - Cocktail-Mix: {entry['cocktail_mix']}")
        print(f"  - Bewertung:    {entry['rating']}")
        print()


def questionnaire_and_cocktail_generator(data):
    """
    - Fragebogen => user_profile
    - Frage: Alkoholisch (a) oder nicht (n)?
    - Frage: wieviele Zutaten max. (k)?
    - Filter Zutaten nach alcoholic-Flag
    - Trainiere Autoencoder => rekonstruierte Taste
    - Mische maximal k Zutaten => Ausgabe
    - Anschließend Bewertung abfragen und speichern.
    """
    questions = data["questions"]
    
    # Geschmacks-Profil sammeln (5-dim. Vektor, je nach Daten)
    user_profile = np.zeros(5)
    valid_answers = 0
    
    for q in questions:
        print("\n" + q["question"])
        print("1) " + q["option1_text"])
        print("2) " + q["option2_text"])
        
        ans = input("Deine Wahl (1/2): ")
        if ans == "1":
            user_profile += np.array(q["option1_taste"])
            valid_answers += 1
        elif ans == "2":
            user_profile += np.array(q["option2_taste"])
            valid_answers += 1
        else:
            print("Ungültige Eingabe, überspringe diese Frage.")
    
    if valid_answers > 0:
        user_profile /= valid_answers
    
    # Alkoholisch oder nicht?
    print("\nBevorzugst du eine alkoholische (a) oder nicht-alkoholische (n) Variante?")
    alc_pref = input("(a/n): ").strip().lower()
    if alc_pref not in ["a", "n"]:
        print("Ungültige Eingabe, setze default nicht-alkoholisch.")
        alc_pref = "n"
    
    # Wieviele Zutaten max.?
    print("\nWie viele Zutaten soll der Cocktail maximal haben? (z.B. 3)")
    try:
        k = int(input("Max. Zutaten: "))
        if k <= 0:
            print("Ungültige Eingabe, setze k=1")
            k = 1
    except ValueError:
        print("Ungültige Eingabe, setze k=3")
        k = 3
    
    # Zutaten nach alc_pref filtern
    all_ingredients = data["ingredients"]
    
    if alc_pref == "a":
        # Alkoholisch + nicht-alkoholisch erlauben
        filtered_items = [
            (name, info["taste"])
            for name, info in all_ingredients.items()
        ]
    else:
        # Nur nicht-alkoholische Zutaten
        filtered_items = [
            (name, info["taste"])
            for name, info in all_ingredients.items()
            if info["alcoholic"] is False
        ]
    
    if not filtered_items:
        print("Keine passenden Zutaten gefunden!")
        return
    
    ingredient_names = [x[0] for x in filtered_items]
    ingredient_profiles = np.array([x[1] for x in filtered_items])
    
    # Autoencoder => Rekonstruktion
    autoencoder = build_and_train_autoencoder(ingredient_profiles)
    reconstructed_profile = autoencoder.predict(np.array([user_profile]))[0]
    
    # Mischung berechnen
    total_volume = 200.0
    final_mix = create_mix_profile(
        reconstructed_profile,
        ingredient_profiles,
        ingredient_names,
        k,
        total_volume=total_volume
    )
    
    # Ergebnis ausgeben
    print("\n---------- ERGEBNIS ----------")
    print(f"Dein Nutzerprofil:                    {user_profile}")
    print(f"Rekonstruiertes Profil (Autoencoder): {reconstructed_profile}")
    print(f"\nVariante: {'alkoholisch' if alc_pref=='a' else 'nicht-alkoholisch'}")
    print(f"Mischung aus maximal {k} Zutaten (von {len(filtered_items)} möglichen):")
    
    for ingr, vol in final_mix.items():
        print(f"  - {ingr}: {vol} ml")
    
    print("-----------------------------")
    
    # Bewertung abfragen
    print("\nMöchtest du diesen Cocktail bewerten? (j/n)")
    rate_ans = input().strip().lower()
    if rate_ans == "j":
        print("Bitte gib eine Bewertung zwischen 1 (schlecht) und 5 (sehr gut) ein:")
        try:
            rating = int(input("Deine Bewertung: "))
            if rating < 1 or rating > 5:
                print("Ungültige Bewertung. Es wird keine Bewertung gespeichert.")
            else:
                # Bewertung speichern
                save_rating(user_profile, final_mix, rating)
        except ValueError:
            print("Ungültige Eingabe, es wird keine Bewertung gespeichert.")

##############################################################################
# 6) MENÜ INTEGRIEREN (HAUPTTEIL)
##############################################################################
if __name__ == "__main__":
    data = load_cocktail_data()
    
    while True:
        print("\n--- Willkommen beim Cocktail-Generator ---")
        print("1. Neuen Cocktail erstellen")
        print("2. Bewertungen anzeigen")
        print("3. Rezeptdatenbank erweitern")
        print("4. Beenden")
        choice = input("Wähle eine Option: ")

        if choice == "1":
            # Neuen Cocktail erstellen
            questionnaire_and_cocktail_generator(data)
        elif choice == "2":
            # Bewertungen anzeigen
            show_ratings()
        elif choice == "3":
            # Rezeptdatenbank erweitern (Platzhalter)
            print("Funktion 'Rezeptdatenbank erweitern' ist noch nicht implementiert.")
        elif choice == "4":
            print("Programm wird beendet. Vielen Dank!")
            break
        else:
            print("Ungültige Auswahl. Bitte versuche es erneut.")
