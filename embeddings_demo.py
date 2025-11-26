from sentence_transformers import SentenceTransformer, util

# Charger le modèle (téléchargé automatiquement si pas présent)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Définir les phrases à comparer
s1 = "assurance habitation"
s2 = "contrat d'assurance maison"
s3 = "voiture électrique"

# Générer les embeddings (vecteurs)
e1, e2, e3 = model.encode([s1, s2, s3])

# Calculer les similarités cosinus entre les paires
sim_1_2 = util.cos_sim(e1, e2).item()
sim_1_3 = util.cos_sim(e1, e3).item()
sim_2_3 = util.cos_sim(e2, e3).item()

# Calculer et afficher les similarités cosinus
print("=" * 50)
print("DÉMONSTRATION DES EMBEDDINGS ET SIMILARITÉ COSINUS")
print("=" * 50)

print(f"\nPhrase 1: '{s1}'")
print(f"Phrase 2: '{s2}'")
print(f"Phrase 3: '{s3}'")

print("\n--- Résultats de similarité ---\n")

sim_1_2 = util.cos_sim(e1, e2).item()
sim_1_3 = util.cos_sim(e1, e3).item()
sim_2_3 = util.cos_sim(e2, e3).item()

print(f"Similarité (s1, s2): {sim_1_2:.4f}")
print(f"  → '{s1}' vs '{s2}'")
print(f"  → Interprétation: {'TRÈS PROCHE' if sim_1_2 > 0.7 else 'PROCHE' if sim_1_2 > 0.5 else 'ÉLOIGNÉ'}")

print(f"\nSimilarité (s1, s3): {sim_1_3:.4f}")
print(f"  → '{s1}' vs '{s3}'")
print(f"  → Interprétation: {'TRÈS PROCHE' if sim_1_3 > 0.7 else 'PROCHE' if sim_1_3 > 0.5 else 'ÉLOIGNÉ'}")

print(f"\nSimilarité (s2, s3): {sim_2_3:.4f}")
print(f"  → '{s2}' vs '{s3}'")
print(f"  → Interprétation: {'TRÈS PROCHE' if sim_2_3 > 0.7 else 'PROCHE' if sim_2_3 > 0.5 else 'ÉLOIGNÉ'}")

print("\n" + "=" * 50)
print("EXPLICATION")
print("=" * 50)
print("""
La similarité cosinus mesure l'angle entre deux vecteurs:
- 1.0  = identiques (angle de 0°)
- 0.0  = orthogonaux (aucune relation)
- -1.0 = opposés (angle de 180°)

En pratique pour les textes:
- > 0.7  : Très similaires (même sujet)
- 0.5-0.7: Similaires (sujets liés)
- < 0.5  : Différents (sujets distincts)
""")
