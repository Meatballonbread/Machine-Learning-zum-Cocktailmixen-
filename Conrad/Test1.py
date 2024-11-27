import json
import random

class CocktailGenerator:
    def __init__(self):
        self.rezepte_db = {}
        self.spirituosen_db = {}
        self.load_data()

    def load_data(self):
        # Lade Rezept-Datenbank
        try:
            with open('cocktail_rezepte_db.json', 'r', encoding='utf-8') as file:
                self.rezepte_db = json.load(file)
        except FileNotFoundError:
            print("Datenbank nicht gefunden, eine neue wird erstellt.")
            self.rezepte_db = {}

        # Initialisiere Spirituosen-Datenbank
        self.spirituosen_db = {
            "süß": "Amaretto",
            "süß/bitter": "Aperol",
            "salzig": "Tequila",
            "cremig/süß": "Baileys",
            "wässrig/sauer": "Gin",
            "wässrig/bitter": "Whiskey",
            "fruchtig/mild": "Rum",
            "scharf": "Chili-Wodka"
        }

    def save_data(self):
        with open('cocktail_rezepte_db.json', 'w', encoding='utf-8') as file:
            json.dump(self.rezepte_db, file, indent=4, ensure_ascii=False)

    def befrage_nutzer(self):
        questions = [
            ("Vollmilch oder Zartbitter?", ["Vollmilch", "Zartbitter"]),
            ("Gummibärchen oder Salzbrezeln?", ["Gummibärchen", "Salzbrezeln"]),
            ("Milchshake oder Zitronenlimo?", ["Milchshake", "Zitronenlimo"]),
            ("Kaffee oder Heiße Schokolade?", ["Kaffee", "Heiße Schokolade"]),
            ("Ketchup oder Senf?", ["Ketchup", "Senf"]),
        ]
        selected_answers = []

        for question, options in questions:
            print(f"{question}")
            print(f"1. {options[0]}")
            print(f"2. {options[1]}")
            answer = input("Wähle 1 oder 2: ")
            if answer == "1":
                selected_answers.append(options[0])
            elif answer == "2":
                selected_answers.append(options[1])
            else:
                print("Ungültige Auswahl! Standardmäßig wird die erste Option gewählt.")
                selected_answers.append(options[0])

        alkohol_preferenz = input("Wie viel Alkohol soll der Cocktail enthalten? (0 = Kein Alkohol, 2 = 2 cl, 4 = 4 cl): ")
        try:
            alkohol_volume = int(alkohol_preferenz)
            if alkohol_volume not in [0, 2, 4]:
                print("Ungültige Eingabe, Standardwert (0 cl) wird verwendet.")
                alkohol_volume = 0
        except ValueError:
            print("Ungültige Eingabe, Standardwert (0 cl) wird verwendet.")
            alkohol_volume = 0

        cocktail = self.generiere_cocktail(selected_answers, alkohol_volume)

        print(f"Dein Cocktail: {cocktail}")
        self.sammle_feedback(cocktail)

    def generiere_cocktail(self, selected_answers, alkohol_volume):
        taste_profiles = {
            "Vollmilch": "süß",
            "Zartbitter": "süß/bitter",
            "Gummibärchen": "süß",
            "Salzbrezeln": "salzig",
            "Milchshake": "cremig/süß",
            "Zitronenlimo": "wässrig/sauer",
            "Kaffee": "wässrig/bitter",
            "Heiße Schokolade": "cremig/süß",
            "Ketchup": "fruchtig/mild",
            "Senf": "scharf"
        }

        cocktail = {}
        total_volume = 200 - alkohol_volume  # Gesamtvolumen in ml

        if not selected_answers:
            print("Keine Antworten ausgewählt, ein Standardcocktail wird generiert.")
            selected_answers = ["Vollmilch"]

        for answer in selected_answers:
            profile = taste_profiles.get(answer)
            if profile:
                ingredient = self.spirituosen_db.get(profile, "Unbekannte Zutat")
                volume = total_volume // len(selected_answers)  # Gleichmäßige Verteilung
                cocktail[ingredient] = volume

        if alkohol_volume > 0:
            cocktail['Alkohol'] = alkohol_volume

        return cocktail

    def sammle_feedback(self, cocktail):
        print("Bitte bewerte den Cocktail mit Sternen (1 bis 5): ")
        feedback = input("Deine Bewertung: ")
        try:
            feedback = int(feedback)
            if feedback < 1 or feedback > 5:
                print("Ungültige Bewertung! Es wird als 3 Sterne gewertet.")
                feedback = 3
        except ValueError:
            print("Ungültige Eingabe! Es wird als 3 Sterne gewertet.")
            feedback = 3

        # Speichern der Bewertung im Rezept-Datenbank
        self.speichere_bewertung(cocktail, feedback)

    def speichere_bewertung(self, cocktail, feedback):
        cocktail_tuple = tuple(cocktail.items())
        if str(cocktail_tuple) not in self.rezepte_db:
            self.rezepte_db[str(cocktail_tuple)] = []

        self.rezepte_db[str(cocktail_tuple)].append(feedback)
        self.save_data()

    def auswertung(self):
        print("\nBewertung der Cocktails:")
        for cocktail, bewertungen in self.rezepte_db.items():
            average_feedback = sum(bewertungen) / len(bewertungen)
            print(f"Cocktail: {cocktail} - Durchschnittliche Bewertung: {average_feedback:.2f} Sterne")

    def erweitere_datenbank(self):
        print("\nMöchtest du neue Geschmacksprofile und Zutaten hinzufügen?")
        while True:
            profile = input("Gib ein neues Geschmacksprofil ein (oder 'beenden', um aufzuhören): ")
            if profile.lower() == "beenden":
                break
            ingredient = input(f"Gib die passende Zutat für das Geschmacksprofil '{profile}' ein: ")
            self.spirituosen_db[profile] = ingredient
            print(f"Das Geschmacksprofil '{profile}' mit der Zutat '{ingredient}' wurde hinzugefügt.")
        print("Aktualisierte Spirituosen-Datenbank:")
        print(self.spirituosen_db)

if __name__ == "__main__":
    generator = CocktailGenerator()
    while True:
        print("\n--- Willkommen beim Cocktail-Generator ---")
        print("1. Neuen Cocktail erstellen")
        print("2. Bewertungen anzeigen")
        print("3. Rezeptdatenbank erweitern")
        print("4. Beenden")
        choice = input("Wähle eine Option: ")

        if choice == "1":
            generator.befrage_nutzer()
        elif choice == "2":
            generator.auswertung()
        elif choice == "3":
            generator.erweitere_datenbank()
        elif choice == "4":
            print("Programm wird beendet. Vielen Dank!")
            break
        else:
            print("Ungültige Auswahl. Bitte versuche es erneut.")
