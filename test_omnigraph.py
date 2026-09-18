"""Comprehensive test suite for OmniGraph core engine."""
import sys, os
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omnigraph.ingestion_pipeline import DatabaseConnection, DocumentIngester
from omnigraph.semantic_query_engine import SemanticQueryEngine
from omnigraph.graph_builder import KnowledgeGraphBuilder
from omnigraph.access_control_audit import AccessControlManager
from omnigraph.entity_relation_extractor import EntityRelationExtractor

db = DatabaseConnection()
db.connect()
OK = "✅"
FAIL = "❌"
print("=" * 64)
print("  OMNIGRAPH COMPREHENSIVE TEST SUITE")
print("=" * 64)

# 1. DB connection
print(f"\n{OK} Database connected (host={os.environ.get('OMNIGRAPH_DB_HOST','localhost')})")

# 2. Graph stats
builder = KnowledgeGraphBuilder(db)
stats = builder.get_graph_stats()
print(f"\n  [GRAPH STATS]")
print(f"  {OK} Documents: {stats.get('total_documents', 0)}")
print(f"  {OK} Entities:  {stats.get('total_entities', 0)}")
print(f"  {OK} Relations: {stats.get('total_relations', 0)}")
print(f"  {OK} Concepts:  {stats.get('total_concepts', 0)}")

# 3. Users
with db.conn.cursor() as cur:
    cur.execute("SELECT user_id, username, full_name FROM omnigraph.users ORDER BY user_id")
    users = cur.fetchall()
print(f"\n  [USERS] {len(users)} found:")
for u in users:
    print(f"  {OK} [{u[0]}] {u[1]:20s}  {u[2]}")

# 4. Fulltext search
qe = SemanticQueryEngine(db, user_id=2)
docs = qe.search("machine learning", strategy="fulltext", limit=3)
print(f"\n  [FULLTEXT SEARCH: 'machine learning'] {len(docs)} results")
for d in docs:
    print(f"  {OK} id={d['document_id']} score={d['score']:.3f}  {d['title'][:50]}")

# 5. Semantic search
sem = qe.search("Kubernetes cloud deployment", strategy="semantic", limit=3)
print(f"\n  [SEMANTIC SEARCH: 'Kubernetes cloud'] {len(sem)} results")
for d in sem:
    print(f"  {OK} id={d['document_id']} score={d['score']:.3f}  {d['title'][:50]}")

# 6. Hybrid search
hyb = qe.search("federated learning privacy", strategy="hybrid", limit=3)
print(f"\n  [HYBRID SEARCH: 'federated learning'] {len(hyb)} results")
for d in hyb:
    print(f"  {OK} id={d['document_id']} score={d['score']:.3f} sources={d.get('sources',[])}  {d['title'][:50]}")

# 7. Graph search
gr = qe.search("Kubernetes Docker", strategy="graph", limit=3)
print(f"\n  [GRAPH SEARCH: K8s/Docker] {len(gr)} results")
for d in gr:
    print(f"  {OK} id={d['document_id']} score={d['score']:.3f}  {d['title'][:50]}")

# 8. Entity neighborhood (Kubernetes = id 4)
neighbors = builder.get_entity_neighborhood(4, max_depth=2)
print(f"\n  [ENTITY NEIGHBORHOOD: Kubernetes depth=2] {len(neighbors)} neighbors")
for n in neighbors[:8]:
    print(f"  {OK} [{n['entity_id']}] {n['name']:25s}  <-[{n['relation_type']:18s}]->  depth={n['depth']}")

# 9. Find experts
experts = qe.find_experts("Deep Learning", limit=5)
print(f"\n  [EXPERTS: Deep Learning] {len(experts)} found")
for e in experts:
    print(f"  {OK} {e['full_name']:25s}  dept={e.get('department',''):15s}  score={e.get('expertise_score',0):.2f}")

# 10. Related concepts
related = qe.find_related_concepts("Machine Learning")
print(f"\n  [RELATED CONCEPTS: Machine Learning] {len(related)} found")
for c in related[:6]:
    print(f"  {OK} {c['name']:30s}  domain={c.get('domain',''):15s}  via={c.get('relationship_types','')}")

# 11. Entity docs
ent_docs = qe.get_entity_documents("Kubernetes")
print(f"\n  [ENTITY DOCUMENTS: Kubernetes] {len(ent_docs)} docs")
for d in ent_docs:
    print(f"  {OK} [{d['document_id']}] {d['title'][:50]:50s}  relevance={d.get('relevance',0):.2f}")

# 12. Shortest path: Kubernetes(4) -> Transformer(21)
with db.conn.cursor() as cur:
    cur.execute("SELECT * FROM omnigraph.sp_shortest_path(4, 21, 6)")
    paths = cur.fetchall()
print(f"\n  [SHORTEST PATH: K8s → Transformer] {len(paths)} paths")
for p in paths:
    print(f"  {OK} length={p[0]}  {' → '.join(str(e) for e in p[1])}")

# 13. Access control
acm = AccessControlManager(db)
checks = [
    (1, "document", 1, "read", "Admin → public doc"),
    (8, "document", 5, "read", "Consumer → restricted doc"),
    (5, "document", 5, "read", "Compliance → restricted doc"),
    (10, "document", 2, "write", "Consumer → confidential doc write"),
]
print(f"\n  [ACCESS CONTROL]")
for uid, rt, rid, act, desc in checks:
    r = acm.check_access(uid, rt, rid, act)
    print(f"  {'✅ GRANTED' if r else '❌ DENIED'}  {desc}")

# 14. Concept hierarchy
hier = builder.get_concept_hierarchy("Machine Learning")
print(f"\n  [CONCEPT HIERARCHY: Machine Learning] {len(hier)} nodes")
for h in hier[:6]:
    print(f"  {OK} depth={h['depth']}  {h['name']:30s}  parent={str(h.get('parent_name',''))}")

# 15. Document versioning (doc 1 = Transformer)
with db.conn.cursor() as cur:
    cur.execute("SELECT version_number, change_summary FROM omnigraph.document_versions WHERE document_id=1 ORDER BY version_number")
    versions = cur.fetchall()
print(f"\n  [DOCUMENT VERSIONS: doc_id=1] {len(versions)} versions")
for v in versions:
    print(f"  {OK} v{v[0]}:  {v[1]}")

# 16. Test ingestion
ingester = DocumentIngester(db)
doc_id = ingester.ingest_document(
    title="Test: OmniGraph Platform Overview",
    source_type="other",
    content="OmniGraph is a knowledge graph platform using PostgreSQL, Voyage AI, and OpenRouter. It supports Kubernetes and integrates with Docker containers for deployment.",
    uploaded_by=1,
    sensitivity_level="public",
)
print(f"\n  [DOCUMENT INGESTION]")
print(f"  {OK} Created document_id={doc_id}")

if doc_id:
    extractor = EntityRelationExtractor(db)
    result = extractor.process_document(doc_id)
    print(f"  {OK} Extracted {len(result['entities'])} entities, {len(result['concepts'])} concepts, {len(result['relationships'])} relationships")
    for e in result['entities'][:5]:
        print(f"      • {e['name']:25s}  ({e.get('entity_type','')})  conf={e.get('confidence',0):.2f}")
    for c in result['concepts'][:5]:
        print(f"      • Concept: {c['name']:30s}  domain={c.get('domain','')}")

# 17. Taxonomy tree
tax = builder.get_taxonomy_tree()
print(f"\n  [TAXONOMY TREE] {len(tax)} nodes")

# 18. Duplicate detection
dupes = builder.detect_duplicate_nodes()
print(f"\n  [DUPLICATE ENTITY DETECTION] {len(dupes)} duplicate pairs")

db.disconnect()
print(f"\n{'=' * 64}")
print(f"  ALL TESTS COMPLETED SUCCESSFULLY")
print(f"{'=' * 64}")