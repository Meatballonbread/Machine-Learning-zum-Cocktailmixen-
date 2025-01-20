def questionnaire_and_cocktail_generator():
    # Fragen, Übersetzungen und Spirituosendatenbank
    questions = [
        ["Vollmilch oder Zartbitter?", "Vollmilch", "Zartbitter"],
        ["Gummibärchen oder Salzbrezeln?", "Gummibärchen", "Salzbrezeln"],
        ["Milchshake oder Zitronenlimo?", "Milchshake", "Zitronenlimo"],
        ["Kaffee oder Heiße Schokolade?", "Kaffee", "Heiße Schokolade"],
        ["Ketchup oder Senf?", "Ketchup", "Senf"]
    ]

    translations = {
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

    spirituousen_db = {
        "süß": "Amaretto",
        "süß/bitter": "Aperol",
        "salzig": "Tequila",
        "cremig/süß": "Baileys",
        "wässrig/sauer": "Gin",
        "wässrig/bitter": "Whiskey",
        "fruchtig/mild": "Rum",
        "scharf": "Chili-Wodka"
    }

    taste_profiles = list(translations.values())
    profile_counts = {profile: 0 for profile in taste_profiles}

    # Benutzerantworten sammeln
    selected_answers = []
    for question in questions:
        print(f"{question[0]}\n1. {question[1]}\n2. {question[2]}")
        user_answer = input("Deine Antwort (1 oder 2): ").strip()
        if user_answer == "1":
            selected_answers.append(question[1])
        elif user_answer == "2":
            selected_answers.append(question[2])
        else:
            print("Ungültige Eingabe. Die Frage wird übersprungen.")
            selected_answers.append(None)

    # Alkoholpräferenz abfragen
    print("\nWie viel Alkohol soll der Cocktail enthalten?\n0 = Kein Alkohol\n2 = 2 cl Alkohol\n4 = 4 cl Alkohol")
    alcohol_preference = input("Deine Wahl: ").strip()
    if alcohol_preference == "0":
        alcohol_volume = 0
    elif alcohol_preference == "2":
        alcohol_volume = 20  # in ml
    elif alcohol_preference == "4":
        alcohol_volume = 40  # in ml
    else:
        print("Ungültige Eingabe. Es wird kein Alkohol hinzugefügt.")
        alcohol_volume = 0

    # Geschmacksprofile analysieren
    for answer in selected_answers:
        if answer and answer in translations:
            profile_counts[translations[answer]] += 1

    # Gesamtvolumen und Verteilung
    total_answers = len(selected_answers)
    total_volume = 200  # in ml
    remaining_volume = total_volume - alcohol_volume

    # Rezept erstellen
    recipe = []
    for profile, count in profile_counts.items():
        if count > 0:
            percentage = count / total_answers
            volume = remaining_volume * percentage
            spirituose = spirituousen_db.get(profile, "Keine Zutat verfügbar")
            recipe.append((spirituose, round(volume, 2)))

    # Ergebnisse ausgeben
    print("\nErgebnisse der Umfrage:")
    for i, answer in enumerate(selected_answers, 1):
        profile = translations.get(answer, "Keine Übersetzung verfügbar")
        print(f"Frage {i}: {answer} -> Geschmacksprofil: {profile}")

    print("\nZusammenfassung der Geschmacksprofile:")
    for profile, count in profile_counts.items():
        if count > 0:
            percentage = round((count / total_answers) * 100, 2)
            print(f"{profile}: {percentage}%")

    print("\nCocktail Rezept:")
    for ingredient, volume in recipe:
        print(f"- {ingredient}: {volume} ml")
    print(f"- Alkohol: {alcohol_volume} ml")

if __name__ == "__main__":
    questionnaire_and_cocktail_generator()
