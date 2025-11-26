from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# === ÉTAPE 1 : Extraire le texte du PDF ===
print("=" * 60)
print("INDEXATION D'UN DOCUMENT PDF DANS UNE BASE VECTORIELLE")
print("=" * 60)

pdf_path = "CGU-1.pdf"
print(f"\n📄 Lecture du fichier: {pdf_path}")

reader = PdfReader(pdf_path)
text = "".join(page.extract_text() for page in reader.pages)

print(f"✅ Nombre de pages: {len(reader.pages)}")
print(f"✅ Nombre total de caractères: {len(text)}")
    
# Afficher un extrait du texte
print(f"\n📝 Extrait du texte (500 premiers caractères):")
print("-" * 40)
print(text[:500])
print("-" * 40)

# === ÉTAPE 2 : Découper le texte en chunks ===
print("\n🔪 Découpage du texte en chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Taille max de chaque chunk
    chunk_overlap=100,   # Chevauchement entre chunks pour le contexte
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]  # Priorité de découpage
)

chunks = splitter.split_text(text)

print(f"✅ Nombre de chunks créés: {len(chunks)}")
print(f"✅ Taille moyenne des chunks: {sum(len(c) for c in chunks) // len(chunks)} caractères")

# Afficher quelques exemples de chunks
print("\n📦 Exemples de chunks:")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ({len(chunk)} caractères) ---")
    print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

# === ÉTAPE 3 : Indexer dans ChromaDB ===
print("\n\n🗄️ Indexation dans ChromaDB...")

# Utiliser HuggingFace embeddings (gratuit et local)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Créer la base vectorielle
persist_directory = "./chroma_db_cgu"

# Supprimer l'ancienne base si elle existe
if os.path.exists(persist_directory):
    import shutil
    shutil.rmtree(persist_directory)
    print(f"🗑️ Ancienne base supprimée")

# Créer les métadonnées pour chaque chunk
metadatas = [{"source": pdf_path, "chunk_id": i} for i in range(len(chunks))]

vectordb = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory=persist_directory,
    collection_name="cgu_collection"
)

print(f"✅ Base vectorielle créée dans: {persist_directory}")
print(f"✅ Nombre de documents indexés: {vectordb._collection.count()}")

# === ÉTAPE 4 : Test de recherche ===
print("\n\n🔍 TEST DE RECHERCHE SÉMANTIQUE")
print("=" * 60)

test_queries = [
    "données personnelles",
    "responsabilité de l'utilisateur",
    "résiliation du contrat"
]

for query in test_queries:
    print(f"\n🔎 Recherche: '{query}'")
    results = vectordb.similarity_search_with_score(query, k=2)
    
    for i, (doc, score) in enumerate(results):
        print(f"\n  📌 Résultat {i+1} (score: {score:.4f}):")
        preview = doc.page_content[:150].replace('\n', ' ')
        print(f"     {preview}...")

# === RÉSUMÉ FINAL ===
print("\n\n" + "=" * 60)
print("📊 RÉSUMÉ DE L'INDEXATION")
print("=" * 60)
print(f"""
Document source:     {pdf_path}
Pages extraites:     {len(reader.pages)}
Caractères totaux:   {len(text)}
Chunks créés:        {len(chunks)}
Taille chunk:        500 caractères (overlap: 100)
Documents indexés:   {vectordb._collection.count()}
Base vectorielle:    {persist_directory}
Modèle embedding:    all-MiniLM-L6-v2 (384 dimensions)
""")
