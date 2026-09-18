import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))



from omnigraph.ingestion_pipeline import DatabaseConnection, DocumentIngester
from omnigraph.embedder import generate_embedding, is_available
from omnigraph.config import settings

db = DatabaseConnection()
db.connect()

# Clear old embeddings
with db.conn.cursor() as cur:
    cur.execute("DELETE FROM omnigraph.embeddings WHERE source_type = 'document'")
    db.conn.commit()
print("Cleared old embeddings")

# Re-embed all documents
ingester = DocumentIngester(db)
ok, fail = ingester.reembed_all_documents()
print(f"Re-embedded: {ok} ok, {fail} failed")

# Verify
with db.conn.cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM omnigraph.embeddings WHERE source_type = 'document'")
    count = cur.fetchone()[0]
print(f"Total document embeddings now: {count}")

db.disconnect()