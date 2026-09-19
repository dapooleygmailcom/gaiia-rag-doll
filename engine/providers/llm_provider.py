"""
llm_provider.py — Unified LLM and Embeddings Provider Layer for Gaiia RAG Doll.

Provides a common interface across:
  - Local Mode: Ollama (llama3.1:8b, qwen2.5:14b, nomic-embed-text)
  - Cloud Mode: AWS Bedrock (amazon.nova-micro-v1:0, amazon.nova-pro-v1:0, amazon.titan-embed-text-v2:0)

Auto-selects provider based on environment, with zero regression for local testing.
"""

import abc
import json
import os
import re
from typing import List, Dict, Any, Optional


class BaseLLMProvider(abc.ABC):
    """Abstract base class for LLM generation and embedding extraction."""

    @abc.abstractmethod
    def generate(
        self,
        model_role: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate text completion.
        model_role: 'fast' (for classification/sub-queries/HyDE) or 'reasoning' (for authoritative adjudication).
        model: optional explicit model name override.
        """
        pass

    @abc.abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate normalized vector embedding for the input text."""
        pass

    def classify_and_decompose(self, query: str) -> Dict[str, Any]:
        """
        Classify query intent, extract explicit rule numbers, and decompose into sub-queries.
        Includes pseudo_clause generation for HyDE embedding alignment.
        """
        system_prompt = (
            "You are an authoritative wargame rules classifier and indexer for the Up Front card game.\n"
            "Analyze the user's question and respond with ONLY a valid JSON object with these exact keys:\n"
            "{\n"
            '  "query_type": "direct_rule" | "concept" | "situation" | "scenario" | "comparison" | "variant",\n'
            '  "rule_numbers": ["list of explicit rule numbers mentioned in query e.g. 5.41, 15.2"],\n'
            '  "scenario": "scenario letter if mentioned or null",\n'
            '  "sub_queries": ["2-3 focused sub-queries expanding abbreviations like SL, AFV, CC, MPh, FP, RR"],\n'
            '  "pseudo_clause": "2-sentence authoritative rulebook-style clause that defines or resolves this scenario"\n'
            "}"
        )
        user_prompt = f"Analyze user query: \"{query}\""
        raw = self.generate(
            model_role="fast",
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=384,
        )

        try:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                data = json.loads(match.group(0))
                # Validate query_type
                valid_types = {"direct_rule", "concept", "situation", "scenario", "comparison", "variant"}
                if data.get("query_type") not in valid_types:
                    data["query_type"] = "concept"
                if not isinstance(data.get("rule_numbers"), list):
                    data["rule_numbers"] = []
                if not isinstance(data.get("sub_queries"), list):
                    data["sub_queries"] = []
                return data
        except Exception:
            pass

        # Fallback if parsing fails
        return {
            "query_type": "concept",
            "rule_numbers": [],
            "scenario": None,
            "sub_queries": [query],
            "pseudo_clause": "",
        }


class LocalOllamaProvider(BaseLLMProvider):
    """Local inference provider wrapping Ollama on local machine."""

    def __init__(
        self,
        fast_model: str = "llama3.1:8b",
        reasoning_model: str = "qwen2.5:14b",
        embed_model: str = "nomic-embed-text",
    ):
        self.fast_model = fast_model
        self.reasoning_model = reasoning_model
        self.embed_model = embed_model
        self._ollama = None

    def _get_ollama(self):
        if self._ollama is None:
            import ollama
            self._ollama = ollama
        return self._ollama

    def generate(
        self,
        model_role: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> str:
        client = self._get_ollama()
        target_model = model or (self.reasoning_model if model_role == "reasoning" else self.fast_model)
        effective_tokens = 2500 if (max_tokens == 1024 and model_role == "reasoning") else max_tokens

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        try:
            response = client.generate(
                model=target_model,
                prompt=full_prompt,
                options={"temperature": temperature, "num_predict": effective_tokens},
            )
            return response.get("response", "").strip()
        except Exception as e:
            # If explicit model override was requested and failed, raise so callers can handle fallback
            if model:
                raise e
            # If large reasoning model is not installed locally, fall back to fast model
            if target_model != self.fast_model:
                try:
                    response = client.generate(
                        model=self.fast_model,
                        prompt=full_prompt,
                        options={"temperature": temperature, "num_predict": max_tokens},
                    )
                    return response.get("response", "").strip()
                except Exception:
                    pass
            print(f"[LocalOllamaProvider] Error calling {target_model}: {e}")
            return ""

    def embed(self, text: str) -> List[float]:
        client = self._get_ollama()
        try:
            response = client.embeddings(model=self.embed_model, prompt=text)
            return response.get("embedding", [])
        except Exception as e:
            print(f"[LocalOllamaProvider] Error embedding text: {e}")
            return []


class CloudBedrockProvider(BaseLLMProvider):
    """Cloud inference provider wrapping Amazon Bedrock in AWS."""

    def __init__(
        self,
        region_name: str = "ap-southeast-2",
        fast_model: Optional[str] = None,
        reasoning_model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ):
        self.region_name = region_name or os.environ.get("AWS_REGION", "ap-southeast-2")
        self.fast_model = fast_model or os.environ.get("BEDROCK_FAST_MODEL", "amazon.nova-lite-v1:0")
        self.reasoning_model = reasoning_model or os.environ.get("BEDROCK_REASONING_MODEL", "amazon.nova-pro-v1:0")
        self.embed_model = embed_model or os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
        self._bedrock = None

    def _get_bedrock(self):
        if self._bedrock is None:
            import boto3
            profile = os.environ.get("AWS_PROFILE")
            if profile:
                session = boto3.Session(profile_name=profile, region_name=self.region_name)
                self._bedrock = session.client("bedrock-runtime")
            else:
                self._bedrock = boto3.client("bedrock-runtime", region_name=self.region_name)
        return self._bedrock

    def generate(
        self,
        model_role: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> str:
        client = self._get_bedrock()
        model_id = model or (self.reasoning_model if model_role == "reasoning" else self.fast_model)
        effective_tokens = 2500 if (max_tokens == 1024 and model_role == "reasoning") else max_tokens

        body: Dict[str, Any] = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"temperature": temperature, "max_new_tokens": effective_tokens},
        }
        if system_prompt:
            body["system"] = [{"text": system_prompt}]

        try:
            response = client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            data = json.loads(response["body"].read().decode("utf-8"))
            content = data.get("output", {}).get("message", {}).get("content", [])
            for part in content:
                if "text" in part:
                    return part["text"].strip()
            return ""
        except Exception as e:
            print(f"[CloudBedrockProvider] Error invoking {model_id}: {e}")
            return ""

    def embed(self, text: str) -> List[float]:
        client = self._get_bedrock()
        # Truncate text to safety limit (Titan max is 50,000 chars / 8192 tokens)
        truncated_text = (text or "").strip()[:20000]
        if not truncated_text:
            truncated_text = "rules reference"
        body = {
            "inputText": truncated_text,
            "dimensions": 1024,
            "normalize": True,
        }
        try:
            response = client.invoke_model(
                modelId=self.embed_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            data = json.loads(response["body"].read().decode("utf-8"))
            emb = data.get("embedding", [])
            if emb:
                return emb
        except Exception as e:
            print(f"[CloudBedrockProvider] Error generating Titan embeddings for '{truncated_text[:50]}...': {e}")
        
        # Deterministic fallback vector if Bedrock embedding call fails
        import hashlib, math
        seed = hashlib.sha256(truncated_text.encode("utf-8")).digest()
        vec = []
        for i in range(1024):
            byte_val = seed[i % len(seed)]
            val = math.sin((i + 1) * byte_val)
            vec.append(val)
        norm = sum(x * x for x in vec) ** 0.5
        return [round(x / norm, 6) for x in vec]


def get_llm_provider(provider_type: Optional[str] = None) -> BaseLLMProvider:
    """
    Factory to get the appropriate LLM provider.
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
        return CloudBedrockProvider()
    return LocalOllamaProvider()
