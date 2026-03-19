"""
ArangoDB connection manager and graph schema setup.
Collections: processes, steps, suggestions, erp_modules
Edge collections: has_step, leads_to, triggers_suggestion, belongs_to_module
"""

from arango import ArangoClient
from arango.exceptions import DatabaseCreateError, CollectionCreateError, GraphCreateError
import os
import logging

logger = logging.getLogger(__name__)

# ── Collection names ──────────────────────────────────────────────────────────
COLLECTIONS = {
    "documents": "processes",         # root process documents
    "steps": "process_steps",         # individual process steps
    "suggestions": "automation_suggestions",
    "erp_modules": "erp_modules",
    "erp_relationships": "erp_relationships",  # node collection for relationship data
}

EDGE_COLLECTIONS = {
    "has_step": "has_step",            # process → step
    "step_sequence": "step_sequence",  # step → step (ordering)
    "triggers_suggestion": "triggers_suggestion",  # step → suggestion
    "belongs_to_module": "belongs_to_module",       # process → erp_module
    "module_relation": "module_relation",            # erp_module → erp_module
}

GRAPH_NAME = "process_graph"

GRAPH_EDGE_DEFINITIONS = [
    {
        "edge_collection": EDGE_COLLECTIONS["has_step"],
        "from_vertex_collections": [COLLECTIONS["documents"]],
        "to_vertex_collections": [COLLECTIONS["steps"]],
    },
    {
        "edge_collection": EDGE_COLLECTIONS["step_sequence"],
        "from_vertex_collections": [COLLECTIONS["steps"]],
        "to_vertex_collections": [COLLECTIONS["steps"]],
    },
    {
        "edge_collection": EDGE_COLLECTIONS["triggers_suggestion"],
        "from_vertex_collections": [COLLECTIONS["steps"]],
        "to_vertex_collections": [COLLECTIONS["suggestions"]],
    },
    {
        "edge_collection": EDGE_COLLECTIONS["belongs_to_module"],
        "from_vertex_collections": [COLLECTIONS["documents"]],
        "to_vertex_collections": [COLLECTIONS["erp_modules"]],
    },
    {
        "edge_collection": EDGE_COLLECTIONS["module_relation"],
        "from_vertex_collections": [COLLECTIONS["erp_modules"]],
        "to_vertex_collections": [COLLECTIONS["erp_modules"]],
    },
]


class ArangoDB:
    _instance = None

    def __init__(self):
        self.client = None
        self.db = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._connect()
        return cls._instance

    def _connect(self):
        host = os.getenv("ARANGO_HOST", "https://409940ccdebc.arangodb.cloud:8529")
        db_name = os.getenv("ARANGO_DB", "agent")
        username = os.getenv("ARANGO_USERNAME", "root")
        password = os.getenv("ARANGO_PASSWORD", "HjzEYC4ilPIudYvQRPdR")

        self.client = ArangoClient(hosts=host)
        sys_db = self.client.db("_system", username=username, password=password)

        try:
            if not sys_db.has_database(db_name):
                sys_db.create_database(db_name)
                logger.info(f"Created database: {db_name}")
        except DatabaseCreateError as e:
            logger.warning(f"DB creation warning: {e}")

        self.db = self.client.db(db_name, username=username, password=password)
        self._ensure_schema()

    def _ensure_schema(self):
        """Create collections, edge collections, and named graph if not present."""
        for name in COLLECTIONS.values():
            if not self.db.has_collection(name):
                self.db.create_collection(name)
                logger.info(f"Created collection: {name}")

        for name in EDGE_COLLECTIONS.values():
            if not self.db.has_collection(name):
                self.db.create_collection(name, edge=True)
                logger.info(f"Created edge collection: {name}")

        if not self.db.has_graph(GRAPH_NAME):
            try:
                self.db.create_graph(
                    GRAPH_NAME,
                    edge_definitions=GRAPH_EDGE_DEFINITIONS
                )
                logger.info(f"Created graph: {GRAPH_NAME}")
            except GraphCreateError as e:
                logger.warning(f"Graph creation warning: {e}")

    def collection(self, name):
        return self.db.collection(name)

    def graph(self):
        return self.db.graph(GRAPH_NAME)

    def aql(self, query, bind_vars=None):
        return self.db.aql.execute(query, bind_vars=bind_vars or {})


def get_db() -> ArangoDB:
    return ArangoDB.get_instance()

def get_graph_context(process_key: str):
    db = get_db()

    # Get steps
    steps = list(db.aql(
        "FOR s IN process_steps FILTER s.process_key == @key RETURN s",
        {"key": process_key}
    ))

    # Get module relationships
    module_rels = list(db.aql(
        """
        FOR e IN module_relation
        FILTER e._from LIKE CONCAT('erp_modules/', '%')
        RETURN e
        """
    ))

    return {
        "steps": steps,
        "relationships": module_rels
    }    
