
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

Die Implemntierung von Supervides Learning ist noch in Überlegung 

Author: Florian, Conrad
Date: 20.01.2025
"""




import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

class CocktailGenerator:
    def __init__(self):
        self.rezepte_db = {}
        self.spirituosen_db = {
            "süß": "Amaretto",
            "bitter": "Aperol",
            "salzig": "Tequila",
            "sauer": "Gin",
            "scharf": "Chili-Wodka"
        }
        self.load_data()

    def load_data(self):
        try:
            with open('cocktail_rezepte_db.json', 'r', encoding='utf-8') as file:
                self.rezepte_db = json.load(file)
        except FileNotFoundError:
            self.rezepte_db = {}

    def save_data(self):
        with open('cocktail_rezepte_db.json', 'w', encoding='utf-8') as file:
            json.dump(self.rezepte_db, file, indent=4, ensure_ascii=False)

    def load_cocktail_data(self):
        """
        Lädt die Cocktail-Daten (Zutaten + Fragen) aus der JSON-Datei.
        """
        json_path = "code_final/cocktail_data_quest.json"
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_and_train_autoencoder(self, ingredient_profiles, latent_dim=3, epochs=500, batch_size=4):
        """
        Erstellt einen Autoencoder und trainiert ihn auf den übergebenen Profilen.
        """
        input_dim = ingredient_profiles.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(latent_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(ingredient_profiles, ingredient_profiles, epochs=epochs, batch_size=batch_size, verbose=0)
        return autoencoder

    def create_mix_profile(self, reconstructed_profile, ingredient_profiles, ingredient_names, k, total_volume=200.0):
        """
        Berechnet die optimale Mischung aus Zutaten basierend auf Least Squares.
        """
        A = ingredient_profiles.T
        b = reconstructed_profile
        w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        w = np.maximum(w, 0)
        sum_w = np.sum(w)
        w = w / sum_w if sum_w > 0 else np.ones_like(w) / len(w)
        if k < len(w):
            sorted_idx = np.argsort(-w)
            top_k_idx = sorted_idx[:k]
            mask = np.zeros_like(w, dtype=bool)
            mask[top_k_idx] = True
            w[~mask] = 0
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w[mask]) / k
        volumes = w * total_volume
        return {name: round(float(vol), 2) for name, vol in zip(ingredient_names, volumes) if vol > 1e-6}

    def questionnaire_and_cocktail_generator(self, data):
        """
        Erzeugt einen Cocktail basierend auf den Nutzereingaben und speichert ihn.
        """
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
        user_profile /= valid_answers if valid_answers > 0 else 1

        alc_pref = input("\nBevorzugst du eine alkoholische (a) oder nicht-alkoholische (n) Variante? (a/n): ").strip().lower()
        alc_pref = alc_pref if alc_pref in ["a", "n"] else "n"

        try:
            k = int(input("\nWie viele Zutaten soll der Cocktail maximal haben? (z.B. 3): "))
            k = max(1, k)
        except ValueError:
            k = 3

        all_ingredients = data["ingredients"]
        filtered_items = [(name, info["taste"]) for name, info in all_ingredients.items() if info["alcoholic"] == (alc_pref == "a")]

        if not filtered_items:
            print("Keine passenden Zutaten gefunden!")
            return

        ingredient_names = [x[0] for x in filtered_items]
        ingredient_profiles = np.array([x[1] for x in filtered_items])
        autoencoder = self.build_and_train_autoencoder(ingredient_profiles)
        reconstructed_profile = autoencoder.predict(np.array([user_profile]))[0]
        final_mix = self.create_mix_profile(reconstructed_profile, ingredient_profiles, ingredient_names, k)

        print("\n---------- ERGEBNIS ----------")
        print(f"Dein Nutzerprofil: {user_profile}")
        print(f"Rekonstruiertes Profil: {reconstructed_profile}")
        print(f"{'Alkoholisch' if alc_pref == 'a' else 'Nicht-alkoholisch'} Cocktail:")
        for ingr, vol in final_mix.items():
            print(f"  - {ingr}: {vol} ml")
        self.sammle_feedback(final_mix)

    def sammle_feedback(self, cocktail):
        feedback = input("\nBitte bewerte den Cocktail mit Sternen (1 bis 5): ")
        try:
            feedback = int(feedback)
            feedback = feedback if 1 <= feedback <= 5 else 3
        except ValueError:
            feedback = 3
        self.speichere_bewertung(cocktail, feedback)

    def speichere_bewertung(self, cocktail, feedback):
        cocktail_tuple = tuple(cocktail.items())
        if str(cocktail_tuple) not in self.rezepte_db:
            self.rezepte_db[str(cocktail_tuple)] = []
        self.rezepte_db[str(cocktail_tuple)].append(feedback)
        self.save_data()

    def erweitere_datenbank(self):
        while True:
            profile = input("Gib ein neues Geschmacksprofil ein (oder 'beenden'): ")
            if profile.lower() == "beenden":
                break
            ingredient = input(f"Gib die passende Zutat für '{profile}' ein: ")
            self.spirituosen_db[profile] = ingredient

    def auswertung(self):
        print("\nBewertungen der Cocktails:")
        for cocktail, bewertungen in self.rezepte_db.items():
            avg_feedback = sum(bewertungen) / len(bewertungen)
            print(f"{cocktail}: Durchschnittliche Bewertung: {avg_feedback:.2f}")

if __name__ == "__main__":
    generator = CocktailGenerator()
    data = generator.load_cocktail_data()

    while True:
        print("\n--- Cocktail-Generator ---")
        print("1. Neuen Cocktail erstellen")
        print("2. Bewertungen anzeigen")
        print("3. Rezeptdatenbank erweitern")
        print("4. Beenden")
        choice = input("Wähle eine Option: ")

        if choice == "1":
            generator.questionnaire_and_cocktail_generator(data)
        elif choice == "2":
            generator.auswertung()
        elif choice == "3":
            generator.erweitere_datenbank()
        elif choice == "4":
            break
        else:
            print("Ungültige Auswahl. Bitte erneut versuchen.")
