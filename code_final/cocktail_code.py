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

Wenn alkoholisch gewählt wird, wird festgelegt, dass 30% des Getränkes aus alkoholhaltigen Zutaten bestehen.

Die Implemntierung von Supervides Learning ist noch in Überlegung 
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
    Berechnet per Least Squares ein Gewichtungsprofil der Zutaten und wählt
    anschließend nur die Top k Zutaten aus, normalisiert die Gewichte und
    rechnet sie auf ml um.
    
    Rückgabe: dict {Zutat: ml} mit genau k Zutaten (eventuell werden Zutaten
              mit 0 ml hinzugefügt, wenn weniger als k Zutaten aktiv waren).
    """
    A = ingredient_profiles.T  # shape (features, M)
    b = reconstructed_profile
    
    # Unbeschränktes Least-Squares
    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Negative Gewichte auf 0 setzen
    w = np.maximum(w, 0)
    
    # Normalisieren => sum(w) = 1 (falls möglich)
    sum_w = np.sum(w)
    if sum_w == 0:
        w = np.ones_like(w) / len(w)
    else:
        w /= sum_w
    
    # Nur Top k Gewichte behalten
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

    # Berechne Volumen in ml
    volumes = w * total_volume
    
    mixture = {}
    for name, vol in zip(ingredient_names, volumes):
        if vol > 1e-6:
            mixture[name] = round(float(vol), 2)
    
    # Falls weniger als k Zutaten aktiv sind, ergänzen wir mit 0 ml
    if len(mixture) < k:
        remaining_ingredients = [name for name in ingredient_names if name not in mixture]
        for i in range(k - len(mixture)):
            if remaining_ingredients:
                ingredient = remaining_ingredients.pop(0)
                mixture[ingredient] = 0.0
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
    - Erfasst das Geschmacksprofil des Nutzers
    - Fragt nach bevorzugter Variante (alkoholisch/nicht-alkoholisch)
    - Fragt nach der gewünschten Anzahl k Zutaten
    - Bei nicht-alkoholischer Variante: Wähle k nicht-alkoholische Zutaten.
    - Bei alkoholischer Variante: Teile k in zwei Gruppen (z. B. 30% alkoholisch, 70% nicht-alkoholisch)
      und berechne die Mischungen separat, sodass insgesamt genau k Zutaten ausgewählt werden.
    """
    # 4.1) Geschmacks-Profil sammeln
    questions = data["questions"]
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
    
    total_volume = 200.0
    all_ingredients = data["ingredients"]
    
    if alc_pref == "n":
        # Nur nicht-alkoholische Zutaten verwenden
        filtered_items = [(name, info["taste"]) for name, info in all_ingredients.items() if not info["alcoholic"]]
        if not filtered_items:
            print("Keine passenden Zutaten gefunden!")
            return
        ingredient_names = [x[0] for x in filtered_items]
        ingredient_profiles = np.array([x[1] for x in filtered_items])
        autoencoder = build_and_train_autoencoder(ingredient_profiles)
        reconstructed_profile = autoencoder.predict(np.array([user_profile]))[0]
        final_mix = create_mix_profile(reconstructed_profile, ingredient_profiles, ingredient_names, k, total_volume=total_volume)
    
    else:
        # Alkoholische Variante: Wir teilen k in zwei Gruppen auf:
        # z.B. 30% der Zutaten (mindestens 1) sollen alkoholisch sein, der Rest nicht-alkoholisch.
        num_alc = max(1, round(0.3 * k))
        num_non_alc = k - num_alc
        
        # Filter Zutaten
        alcoholic_items = [(name, info["taste"]) for name, info in all_ingredients.items() if info["alcoholic"]]
        non_alcoholic_items = [(name, info["taste"]) for name, info in all_ingredients.items() if not info["alcoholic"]]
        
        # Falls in einer Gruppe nicht genügend Zutaten vorhanden sind, passen wir die Verteilung an.
        if len(alcoholic_items) < num_alc:
            num_alc = len(alcoholic_items)
            num_non_alc = k - num_alc
        if len(non_alcoholic_items) < num_non_alc:
            num_non_alc = len(non_alcoholic_items)
            num_alc = k - num_non_alc
        
        # Alkoholische Gruppe
        alc_names = [x[0] for x in alcoholic_items]
        alc_profiles = np.array([x[1] for x in alcoholic_items])
        autoencoder_alc = build_and_train_autoencoder(alc_profiles)
        reconstructed_profile_alc = autoencoder_alc.predict(np.array([user_profile]))[0]
        final_mix_alc = create_mix_profile(reconstructed_profile_alc, alc_profiles, alc_names, num_alc, total_volume=total_volume * 0.3)
        
        # Nicht-alkoholische Gruppe
        non_alc_names = [x[0] for x in non_alcoholic_items]
        non_alc_profiles = np.array([x[1] for x in non_alcoholic_items])
        autoencoder_non_alc = build_and_train_autoencoder(non_alc_profiles)
        reconstructed_profile_non_alc = autoencoder_non_alc.predict(np.array([user_profile]))[0]
        final_mix_non_alc = create_mix_profile(reconstructed_profile_non_alc, non_alc_profiles, non_alc_names, num_non_alc, total_volume=total_volume * 0.7)
        
        # Beide Gruppen kombinieren
        final_mix = {**final_mix_alc, **final_mix_non_alc}
    
    # Ergebnis ausgeben
    print("\n---------- ERGEBNIS ----------")
    print(f"Dein Nutzerprofil:                   {user_profile}")
    if alc_pref == "n":
        print(f"Rekonstruiertes Profil (Autoencoder): {reconstructed_profile}")
    else:
        print(f"Rekonstruiertes Profil (alkoholisch): {reconstructed_profile_alc}")
        print(f"Rekonstruiertes Profil (nicht-alkoholisch): {reconstructed_profile_non_alc}")
    print(f"\nVariante: {'alkoholisch' if alc_pref=='a' else 'nicht-alkoholisch'}")
    print(f"Mischung aus genau {k} Zutaten:")
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
