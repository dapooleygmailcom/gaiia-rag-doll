"""
storage_provider.py — Unified Storage and Knowledge Base Provider Layer for Gaiia RAG Doll.

Provides a common interface across:
  - Local Mode: ChromaDB + local JSON rule indices & hierarchy graphs.
  - Cloud Mode: AWS DynamoDB (RagDoll-Knowledge-test) + S3 vector index assets (vector_index.json).

Auto-selects provider based on environment, with zero regression for local testing.
"""

import abc
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple


class BaseStorageProvider(abc.ABC):
    """Abstract base class for rule knowledge retrieval and storage."""

    @abc.abstractmethod
    def get_rule(self, title_id: str, rule_number: str) -> Optional[Dict[str, Any]]:
        """Retrieve authoritative codified rule record by rule number."""
        pass

    @abc.abstractmethod
    def search_vectors(
        self,
        title_id: str,
        query_vector: List[float],
        keywords: List[str],
        top_n: int = 14,
    ) -> List[str]:
        """Perform hybrid vector and keyword search, returning ranked rule numbers."""
        pass

    @abc.abstractmethod
    def get_section_tree(self, title_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve hierarchical section tree for section sibling expansion."""
        pass

    @abc.abstractmethod
    def get_cooccurrence_graph(self, title_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve 2-hop co-occurrence graph for concept expansion."""
        pass


class LocalChromaStorageProvider(BaseStorageProvider):
    """Local storage provider wrapping ChromaDB and local JSON files."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = project_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../..")
        )
        self.chroma_dir = os.path.join(self.project_root, "data/chroma")
        self.collection_name = "upfront-rules-semantic"
        self.rule_index_file = os.path.join(self.project_root, "data/upfront_rule_index.json")
        self.section_tree_file = os.path.join(self.project_root, "data/upfront_section_tree.json")
        self.cooc_graph_file = os.path.join(self.project_root, "data/upfront_cooccurrence_graph.json")

        self._collection = None
        self._rule_index = None
        self._section_tree = None
        self._cooc_graph = None

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=self.chroma_dir)
            self._collection = client.get_collection(name=self.collection_name)
        return self._collection

    def _get_rule_index(self) -> Dict[str, Any]:
        if self._rule_index is None:
            if os.path.exists(self.rule_index_file):
                with open(self.rule_index_file, "r", encoding="utf-8") as f:
                    self._rule_index = json.load(f)
            else:
                self._rule_index = {}
        return self._rule_index

    def get_rule(self, title_id: str, rule_number: str) -> Optional[Dict[str, Any]]:
        rule_idx = self._get_rule_index()
        entries = rule_idx.get(rule_number, [])
        if not entries:
            return None

        # Fetch chunk from ChromaDB
        chunk_id = entries[0]["chunk_id"]
        try:
            col = self._get_collection()
            res = col.get(ids=[chunk_id], include=["documents", "metadatas"])
            docs = res.get("documents", [])
            metas = res.get("metadatas", [])
            if docs and metas:
                return {
                    "ruleNumber": rule_number,
                    "title": metas[0].get("title", f"Rule {rule_number}"),
                    "verbatimText": docs[0],
                    "chapter": metas[0].get("root_section", "Core Rules"),
                    "crossReferences": metas[0].get("cross_references", []),
                    "breadcrumbs": metas[0].get("breadcrumbs", []),
                }
        except Exception as e:
            print(f"[LocalChromaStorageProvider] Error fetching rule {rule_number}: {e}")
        return None

    def search_vectors(
        self,
        title_id: str,
        query_vector: List[float],
        keywords: List[str],
        top_n: int = 14,
    ) -> List[str]:
        if not query_vector:
            return []

        col = self._get_collection()
        oversample_n = int(top_n * 2.5)
        try:
            res = col.query(
                query_embeddings=[query_vector],
                n_results=oversample_n,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as e:
            print(f"[LocalChromaStorageProvider] Chroma query error: {e}")
            return []

        if not res or not res.get("ids") or len(res["ids"][0]) == 0:
            return []

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        scored = []
        for i in range(len(docs)):
            text = docs[i].lower()
            match_count = sum(1 for kw in keywords if kw.lower() in text)
            boost = match_count * 0.12
            final_score = dists[i] - boost

            rn = metas[i].get("rule_number") or ""
            if rn:
                scored.append((rn, final_score))

        scored.sort(key=lambda x: x[1])
        unique_rules = []
        for rn, _ in scored:
            if rn not in unique_rules:
                unique_rules.append(rn)
            if len(unique_rules) >= top_n:
                break
        return unique_rules

    def get_section_tree(self, title_id: str) -> Optional[Dict[str, Any]]:
        if self._section_tree is None:
            if os.path.exists(self.section_tree_file):
                with open(self.section_tree_file, "r", encoding="utf-8") as f:
                    self._section_tree = json.load(f)
        return self._section_tree

    def get_cooccurrence_graph(self, title_id: str) -> Optional[Dict[str, Any]]:
        if self._cooc_graph is None:
            if os.path.exists(self.cooc_graph_file):
                with open(self.cooc_graph_file, "r", encoding="utf-8") as f:
                    self._cooc_graph = json.load(f)
        return self._cooc_graph


class CloudDynamoStorageProvider(BaseStorageProvider):
    """Cloud storage provider wrapping DynamoDB and S3 cached vector assets."""

    def __init__(
        self,
        knowledge_table: Optional[str] = None,
        documents_bucket: Optional[str] = None,
        region_name: str = "ap-southeast-2",
        tenant_id: Optional[str] = None,
    ):
        self.knowledge_table = knowledge_table or os.environ.get("KNOWLEDGE_TABLE_NAME", "RagDoll-Knowledge-test")
        self.documents_bucket = documents_bucket or os.environ.get("DOCUMENTS_BUCKET_NAME", "ragdoll-docs-test-1860")
        self.region_name = region_name or os.environ.get("AWS_REGION", "ap-southeast-2")
        self.tenant_id = tenant_id

        self._dynamodb = None
        self._s3 = None
        self._vector_index_cache: Dict[str, Any] = {}

    @staticmethod
    def _normalize_title_id(title_id: str) -> str:
        tid = str(title_id).replace("_", "-").lower().strip()
        if tid in {"up-front", "upfront"}:
            return "up-front-core"
        return tid

    def _get_dynamo_table(self):
        if self._dynamodb is None:
            import boto3
            profile = os.environ.get("AWS_PROFILE")
            if profile:
                session = boto3.Session(profile_name=profile, region_name=self.region_name)
                ddb = session.resource("dynamodb")
            else:
                ddb = boto3.resource("dynamodb", region_name=self.region_name)
            self._dynamodb = ddb.Table(self.knowledge_table)
        return self._dynamodb

    def _get_s3(self):
        if self._s3 is None:
            import boto3
            profile = os.environ.get("AWS_PROFILE")
            if profile:
                session = boto3.Session(profile_name=profile, region_name=self.region_name)
                self._s3 = session.client("s3")
            else:
                self._s3 = boto3.client("s3", region_name=self.region_name)
        return self._s3

    def _get_vector_index(self, title_id: str, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        norm_title = self._normalize_title_id(title_id)
        if norm_title in self._vector_index_cache:
            return self._vector_index_cache[norm_title]

        candidate_ids = [norm_title]
        if not norm_title.endswith("-core"):
            candidate_ids.append(f"{norm_title}-core")
        elif norm_title.endswith("-core"):
            candidate_ids.append(norm_title[:-5])

        # 1. Try local asset path first (e.g. within bundle or repository)
        backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        for cid in candidate_ids:
            candidate_paths = [
                os.path.join(backend_root, f"assets/titles/{cid}/vector_index.json"),
                f"/var/task/assets/titles/{cid}/vector_index.json",
                os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../../gaiia-rag-doll-cloud/backend/assets/titles/{cid}/vector_index.json")),
            ]
            for local_asset_path in candidate_paths:
                if os.path.exists(local_asset_path):
                    try:
                        with open(local_asset_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            self._vector_index_cache[norm_title] = data
                            return data
                    except Exception as e:
                        print(f"[CloudDynamoStorageProvider] Local asset load notice for {local_asset_path}: {e}")

        # 2. Try S3 bucket fetch if tenant_id is provided
        resolved_tenant_id = tenant_id or self.tenant_id or "internal-test-tenant"
        if resolved_tenant_id:
            s3_client = self._get_s3()
            for cid in candidate_ids:
                s3_key = f"tenants/{resolved_tenant_id}/titles/{cid}/vector_index.json"
                try:
                    res = s3_client.get_object(Bucket=self.documents_bucket, Key=s3_key)
                    body_str = res["Body"].read().decode("utf-8")
                    data = json.loads(body_str)
                    self._vector_index_cache[norm_title] = data
                    return data
                except Exception:
                    pass

        print(f"[CloudDynamoStorageProvider] S3 index notice: Vector index not found for {norm_title} (checked {candidate_ids})")
        return None

    def get_rule_index(self, title_id: str) -> Dict[str, Any]:
        norm_title = self._normalize_title_id(title_id)
        index_data = self._get_vector_index(norm_title)
        rule_index = {}
        if index_data and "rulesMeta" in index_data:
            for rn, meta in index_data["rulesMeta"].items():
                rule_index[rn] = [{
                    "chunk_id": rn,
                    "rule_number": rn,
                    "priority": 1,
                    "metadata": meta,
                }]

        # Also merge local rule index if specifically present for this title
        for cand_id in [norm_title, title_id]:
            local_idx_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), f"../../data/{cand_id}_rule_index.json")
            )
            if os.path.exists(local_idx_path):
                try:
                    with open(local_idx_path, "r", encoding="utf-8") as f:
                        local_idx = json.load(f)
                        for k, v in local_idx.items():
                            if k not in rule_index:
                                rule_index[k] = v
                except Exception:
                    pass
        return rule_index

    def get_rule(self, title_id: str, rule_number: str) -> Optional[Dict[str, Any]]:
        norm_title = self._normalize_title_id(title_id)
        rn = str(rule_number).strip()
        if rn.startswith("RULE#"):
            rn = rn[5:]
        tbl = self._get_dynamo_table()

        candidate_pks = [f"TITLE#{norm_title}"]
        if not norm_title.endswith("-core"):
            candidate_pks.append(f"TITLE#{norm_title}-core")
        elif norm_title.endswith("-core"):
            candidate_pks.append(f"TITLE#{norm_title[:-5]}")

        for pk in candidate_pks:
            try:
                res = tbl.get_item(
                    Key={
                        "PK": pk,
                        "SK": f"RULE#{rn}",
                    }
                )
                if res.get("Item"):
                    return res["Item"]
            except Exception as e:
                print(f"[CloudDynamoStorageProvider] DynamoDB get_item notice for {pk}/{rn}: {e}")
        return None

    def search_vectors(
        self,
        title_id: str,
        query_vector: List[float],
        keywords: List[str],
        top_n: int = 14,
    ) -> List[str]:
        norm_title = self._normalize_title_id(title_id)
        index_data = self._get_vector_index(norm_title)
        if not index_data:
            return []

        vectors: Dict[str, List[float]] = index_data.get("vectors", {})
        title_index: Dict[str, List[str]] = index_data.get("titleIndex", {})
        keyword_index: Dict[str, List[str]] = index_data.get("keywordIndex", {})

        scored = []
        has_vec = bool(query_vector and len(query_vector) > 0)

        for rule_number, vec in vectors.items():
            # 1. Cosine similarity via dot product
            dot = 0.0
            if has_vec:
                length = min(len(query_vector), len(vec))
                for i in range(length):
                    dot += query_vector[i] * vec[i]

            # 2. Two-tier keyword boost
            boost = 0.0
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in title_index and rule_number in title_index[kw_lower]:
                    boost += 1.5
                elif kw_lower in keyword_index and rule_number in keyword_index[kw_lower]:
                    boost += 0.3

            final_score = dot + boost
            scored.append((rule_number, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [rn for rn, _ in scored[:top_n]]

    def get_section_tree(self, title_id: str) -> Optional[Dict[str, Any]]:
        index_data = self._get_vector_index(title_id)
        if index_data and "sectionTree" in index_data:
            return index_data["sectionTree"]
        norm_title = self._normalize_title_id(title_id)
        backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        candidates = [
            os.path.join(backend_root, f"data/{norm_title}_section_tree.json"),
            f"/var/task/data/{norm_title}_section_tree.json",
        ]
        if norm_title == "up-front-core":
            candidates.append(os.path.join(backend_root, "data/up_front_section_tree.json"))
            candidates.append("/var/task/data/up_front_section_tree.json")

        for candidate_path in candidates:
            if os.path.exists(candidate_path):
                try:
                    with open(candidate_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
        return None

    def get_cooccurrence_graph(self, title_id: str) -> Optional[Dict[str, Any]]:
        index_data = self._get_vector_index(title_id)
        if index_data and "cooccurrenceGraph" in index_data:
            return index_data["cooccurrenceGraph"]
        norm_title = self._normalize_title_id(title_id)
        backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        candidates = [
            os.path.join(backend_root, f"data/{norm_title}_cooccurrence_graph.json"),
            f"/var/task/data/{norm_title}_cooccurrence_graph.json",
        ]
        if norm_title == "up-front-core":
            candidates.append(os.path.join(backend_root, "data/up_front_cooccurrence_graph.json"))
            candidates.append("/var/task/data/up_front_cooccurrence_graph.json")

        for candidate_path in candidates:
            if os.path.exists(candidate_path):
                try:
                    with open(candidate_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
        return None


class StorageCollectionAdapter:
    """
    Adapter that mimics ChromaDB collection.get(ids=...) interface,
    routing chunk and rule lookups seamlessly to BaseStorageProvider.
    Ensures identical execution paths for hierarchical/graph expansion stages across Local and Cloud.
    """
    def __init__(self, storage_provider: BaseStorageProvider, title_id: str):
        self.storage = storage_provider
        self.title_id = title_id
        self._chunk_to_rule_map: Optional[Dict[str, str]] = None

    def _get_chunk_to_rule_map(self) -> Dict[str, str]:
        if self._chunk_to_rule_map is None:
            mapping = {}
            if hasattr(self.storage, "get_rule_index"):
                try:
                    rule_idx = self.storage.get_rule_index(self.title_id)
                    for rn, entries in rule_idx.items():
                        if isinstance(entries, list):
                            for entry in entries:
                                cid = entry.get("chunk_id")
                                if cid:
                                    mapping[cid] = str(rn)
                                    mapping[cid.lower()] = str(rn)
                except Exception as e:
                    print(f"[StorageCollectionAdapter] Notice: Rule index map init: {e}")
            self._chunk_to_rule_map = mapping
        return self._chunk_to_rule_map

    def get(self, ids: List[str], include: Optional[List[str]] = None) -> Dict[str, List[Any]]:
        docs, metas = [], []
        seen = set()
        chunk_map = self._get_chunk_to_rule_map()

        for rule_or_id in ids:
            rn = str(rule_or_id).strip()
            if rn.startswith("RULE#"):
                rn = rn[5:]

            # Resolve chunk_id to canonical rule number if needed
            if rn in chunk_map:
                rn = chunk_map[rn]
            elif rn.lower() in chunk_map:
                rn = chunk_map[rn.lower()]

            if rn in seen:
                continue
            seen.add(rn)
            item = self.storage.get_rule(self.title_id, rn)
            if item:
                source_files = item.get("sourceFiles", [item.get("chapter", self.title_id)])
                src = source_files[0] if isinstance(source_files, list) and source_files else item.get("chapter", self.title_id)
                docs.append(item.get("verbatimText", ""))
                metas.append({
                    "rule_number": item.get("ruleNumber", rn),
                    "source_file": src,
                    "priority": int(item.get("priority", 1)),
                    "title": item.get("title", f"Rule {rn}"),
                    "chapter": item.get("chapter", ""),
                    "breadcrumbs": item.get("breadcrumbs", []),
                    "cross_references": item.get("crossReferences", []),
                })
        return {"documents": docs, "metadatas": metas}


def get_storage_provider(provider_type: Optional[str] = None) -> BaseStorageProvider:
    """
    Factory to get the appropriate Storage provider.
    Priority:
      1. Explicit argument ('local' or 'cloud')
      2. Environment variable: PROVIDER, TARGET, or RAGDOLL_TARGET
      3. Auto-detected AWS execution environment (AWS_LAMBDA_FUNCTION_NAME)
      4. Default: 'local' (preserves 100% backward compatibility for existing tests)
    """
    selected = provider_type or os.environ.get("PROVIDER") or os.environ.get("TARGET") or os.environ.get("RAGDOLL_TARGET")
    if not selected and os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        selected = "cloud"

    if selected and selected.lower() == "cloud":
        return CloudDynamoStorageProvider()
    return LocalChromaStorageProvider()
