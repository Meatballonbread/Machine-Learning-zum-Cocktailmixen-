import json
import random
import matplotlib.pyplot as plt

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
        alkohol_volume = int(alkohol_preferenz) if alkohol_preferenz in ['0', '2', '4'] else 0
        cocktail = self.generiere_cocktail(selected_answers, alkohol_volume)

        print(f"Dein Cocktail: {cocktail}")
        self.sammle_feedback(cocktail)

    def generiere_cocktail(self, selected_answers, alkohol_volume):
        taste_profiles = {
            "Vollmilch": "süß",
            "Zartbitter": "bitter",
            "Gummibärchen": "süß",
            "Salzbrezeln": "salzig",
            "Milchshake": "süß",
            "Zitronenlimo": "sauer",
            "Kaffee": "bitter",
            "Heiße Schokolade": "süß",
            "Ketchup": "sauer",
            "Senf": "scharf"
        }

        cocktail = {}
        total_volume = 200 - alkohol_volume  # Gesamtvolumen in ml

        for answer in selected_answers:
            profile = taste_profiles.get(answer)
            if profile:
                ingredient = self.spirituosen_db.get(profile)
                if ingredient:
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

        self.speichere_bewertung(cocktail, feedback)
        if feedback < 3:  # Wenn die Bewertung schlecht ist, optimiere das Rezept
            print("Das Rezept wird aufgrund der schlechten Bewertung angepasst.")
            self.optimiere_rezept(cocktail)
        else:
            print("Das Rezept bleibt unverändert.")

    def speichere_bewertung(self, cocktail, feedback):
        cocktail_tuple = tuple(cocktail.items())
        if str(cocktail_tuple) not in self.rezepte_db:
            self.rezepte_db[str(cocktail_tuple)] = []

        self.rezepte_db[str(cocktail_tuple)].append(feedback)
        self.save_data()

    def optimiere_rezept(self, cocktail):
        for ingredient in list(cocktail.keys()):  # Originalzutaten durchgehen
            if ingredient in self.spirituosen_db.values():
                # Wähle eine neue Zutat, die nicht die gleiche ist
                new_ingredient = random.choice(list(self.spirituosen_db.values()))
                while new_ingredient == ingredient:
                    new_ingredient = random.choice(list(self.spirituosen_db.values()))

                print(f"Zutat {ingredient} wurde durch {new_ingredient} ersetzt.")
                cocktail[new_ingredient] = cocktail.pop(ingredient)

        print(f"Optimiertes Rezept: {cocktail}")

    def analysiere_modifikationen(self):
        print("\nModifikationen der Rezepte:")
        for original, modifikation in self.rezepte_db.items():
            print(f"Original: {original} -> Modifikationen: {modifikation}")

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

    def auswertung(self):
        print("\nBewertung der Cocktails:")
        for cocktail, bewertungen in self.rezepte_db.items():
            average_feedback = sum(bewertungen) / len(bewertungen)
            print(f"Cocktail: {cocktail} - Durchschnittliche Bewertung: {average_feedback:.2f} Sterne")

if __name__ == "__main__":
    generator = CocktailGenerator()
    while True:
        print("\n--- Willkommen beim Cocktail-Generator ---")
        print("1. Neuen Cocktail erstellen")
        print("2. Bewertungen anzeigen")
        print("3. Rezeptdatenbank erweitern")
        print("4. Modifikationen analysieren")
        print("5. Beenden")
        choice = input("Wähle eine Option: ")

        if choice == "1":
            generator.befrage_nutzer()
