"""
Vertex Gemini Provider for Hermes.
Uses Google GenAI SDK to access Gemini models via Vertex AI.
"""

import json
import logging
import os
import sys
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from agent.anthropic_adapter import normalize_vertex_gcp_region

logger = logging.getLogger(__name__)

# User-facing debug mode via environment variable
HERMES_DEBUG = os.getenv("HERMES_DEBUG", "").lower() in ("1", "true", "yes")

def debug_print(msg: str):
    """Explicitly print to stderr if HERMES_DEBUG is enabled."""
    if HERMES_DEBUG:
        print(f"DEBUG [vertex-gemini]: {msg}", file=sys.stderr)


def _vertex_tool_calls_to_openai_objects(raw_calls: List[Dict[str, Any]]) -> List[SimpleNamespace]:
    """Hermes expects Chat Completions-style objects (.function.name), not dicts."""
    out: List[SimpleNamespace] = []
    for tc in raw_calls:
        fn = tc.get("function") or {}
        out.append(
            SimpleNamespace(
                id=tc.get("id"),
                type=tc.get("type", "function"),
                function=SimpleNamespace(
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", "{}"),
                ),
            )
        )
    return out


def resolve_vertex_gemini_credentials() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve Vertex AI credentials for Gemini from environment.

    Returns:
        Tuple of (project_id, location, model) or (None, None, None) if not configured.
    """
    project = (
        os.environ.get("VERTEX_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT_ID")
    )
    location = os.environ.get("VERTEX_LOCATION") or os.environ.get("GCP_LOCATION", "us-central1")
    location = normalize_vertex_gcp_region(location) or location
    model = os.environ.get("VERTEX_GEMINI_MODEL") or os.environ.get("VERTEX_GEMINI_MODEL_NAME")

    if not project:
        return None, None, None

    return project, location, model


def _format_messages_for_gemini(hermes_messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Any]]:
    """Translate Hermes messages to Gemini format.

    Args:
        hermes_messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Tuple of (system_instruction, gemini_contents)
    """
    try:
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai is required for vertex-gemini provider. "
            "Install with: pip install google-genai"
        )

    system_parts = []
    gemini_contents = []

    # Helper to peek at the next role
    def get_role(idx):
        if idx < len(hermes_messages):
            return hermes_messages[idx].get("role", "")
        return None

    i = 0
    while i < len(hermes_messages):
        msg = hermes_messages[i]
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "system":
            system_parts.append(content)
            i += 1
        elif role == "user":
            gemini_contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)]
                )
            )
            i += 1
        elif role == "assistant":
            # Handle tool calls in assistant response
            if msg.get("tool_calls"):
                parts = []
                if content:
                    parts.append(types.Part.from_text(text=content))
                
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    # Vertex AI forbids hyphens in function names
                    safe_name = func.get("name", "").replace("-", "_")
                    
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                        
                    parts.append(
                        types.Part.from_function_call(
                            name=safe_name,
                            args=args
                        )
                    )
                gemini_contents.append(
                    types.Content(role="model", parts=parts)
                )
            else:
                gemini_contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=content)]
                    )
                )
            i += 1
        elif role == "tool":
            # Group consecutive tool results into a single Gemini content turn
            # Gemini REQUIRES all responses for a parallel call turn to be in one message
            tool_parts = []
            
            while i < len(hermes_messages) and hermes_messages[i].get("role") == "tool":
                curr_msg = hermes_messages[i]
                curr_content = curr_msg.get("content", "") or ""
                
                # Resolve function name
                tool_name = curr_msg.get("name")
                tool_call_id = curr_msg.get("tool_call_id")
                
                if not tool_name and tool_call_id:
                    # Scan backwards for the corresponding assistant tool call
                    for prev_msg in reversed(hermes_messages[:i]):
                        if prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
                            for tc in prev_msg["tool_calls"]:
                                if tc.get("id") == tool_call_id:
                                    tool_name = tc.get("function", {}).get("name")
                                    break
                        if tool_name:
                            break
                
                if not tool_name:
                    tool_name = "unknown_tool"

                safe_name = tool_name.replace("-", "_")
                
                # The tool result must be formatted as a dict for 'response'
                try:
                    result_data = json.loads(curr_content) if isinstance(curr_content, str) and curr_content.strip().startswith("{") else {"result": curr_content}
                except Exception:
                    result_data = {"result": curr_content}

                tool_parts.append(
                    types.Part.from_function_response(
                        name=safe_name,
                        response=result_data
                    )
                )
                i += 1
            
            if tool_parts:
                gemini_contents.append(
                    types.Content(
                        role="user",
                        parts=tool_parts
                    )
                )
        else:
            i += 1

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, gemini_contents


def _format_tools_for_gemini(hermes_tools: Optional[List[Dict]]) -> Tuple[Optional[List[Any]], Dict[str, str]]:
    """Translate Hermes JSON Schema tools to Gemini FunctionDeclarations.

    Uses proper SDK types (types.Tool, types.FunctionDeclaration, types.Schema)
    for perfect serialization and response parsing.

    Args:
        hermes_tools: List of tool definitions in Hermes format

    Returns:
        Tuple of (list of SDK Tool objects, tool_name_map) where tool_name_map
        maps safe_name -> original_name for reverse translation.
    """
    if not hermes_tools:
        return None, {}

    from google.genai import types

    # Pure strings for types → no EnumType.__call__ error
    TYPE_MAP = {
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }

    gemini_tools = []
    tool_name_map = {}  # safe_name (underscores) → original Hermes name (hyphens)

    for tool in hermes_tools:
        if tool.get("type") == "function":
            func_data = tool["function"]

            # Hyphen sanitization (Vertex AI forbids hyphens in function names)
            original_name = func_data["name"]
            safe_name = original_name.replace("-", "_")
            tool_name_map[safe_name] = original_name

            properties = {}
            raw_props = func_data.get("parameters", {}).get("properties", {})

            for prop_name, prop_details in raw_props.items():
                json_type = prop_details.get("type", "string")

                prop_schema = types.Schema(
                    type=TYPE_MAP.get(json_type, "STRING"),
                    description=prop_details.get("description", "")
                )

                # Array items (Hermes uses these a lot)
                if json_type == "array" and "items" in prop_details:
                    item_type = prop_details["items"].get("type", "string")
                    prop_schema.items = types.Schema(
                        type=TYPE_MAP.get(item_type, "STRING")
                    )

                if "format" in prop_details:
                    prop_schema.format = prop_details["format"]
                if "title" in prop_details:
                    prop_schema.title = prop_details["title"]

                properties[prop_name] = prop_schema

            # Official SDK objects → perfect serialization
            func_decl = types.FunctionDeclaration(
                name=safe_name,
                description=func_data.get("description", ""),
                parameters=types.Schema(
                    type="OBJECT",
                    properties=properties,
                    required=func_data.get("parameters", {}).get("required", [])
                )
            )

            gemini_tools.append(
                types.Tool(function_declarations=[func_decl])
            )

    return gemini_tools if gemini_tools else None, tool_name_map


class VertexGeminiProvider:
    """Provider for Gemini models via Google Vertex AI.

    Uses Application Default Credentials (ADC) for authentication.
    Set VERTEX_PROJECT and optionally VERTEX_LOCATION environment variables.
    Model can be set via VERTEX_GEMINI_MODEL env var or constructor.
    """

    def __init__(self, model_name: str = None):
        self.project_id = None
        self.location = None
        self.client = None
        # If model_name not passed, will be resolved from env or use default
        self._env_model = model_name
        self.model_name = model_name or "gemini-2.5-flash"
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the Vertex Gemini client."""
        self.project_id, self.location, env_model = resolve_vertex_gemini_credentials()

        if not self.project_id:
            raise ValueError(
                "VERTEX_PROJECT environment variable is required for vertex-gemini provider. "
                "Alternatively set GOOGLE_CLOUD_PROJECT or GCP_PROJECT_ID."
            )

        # Use env model if no model was passed to constructor
        if env_model and not self._env_model:
            self.model_name = env_model

        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai is required for vertex-gemini provider. "
                "Install with: pip install google-genai"
            )

        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location
        )
        
        # Mimic OpenAI client structure for Hermes compatibility
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.generate
            )
        )

        logger.debug(
            "Vertex Gemini provider initialized: project=%s, location=%s, model=%s",
            self.project_id, self.location, self.model_name
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response using Vertex Gemini.

        Args:
            messages: Hermes-format message history
            tools: Hermes-format tool definitions
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            Response dict in OpenAI envelope format for Hermes compatibility
        """
        from google.genai import types

        system_instruction, contents = _format_messages_for_gemini(messages)
        gemini_tools, tool_name_map = _format_tools_for_gemini(tools)

        if HERMES_DEBUG:
            debug_print(f"Vertex Gemini Request Contents: {len(contents)} messages")
            for i, c in enumerate(contents):
                role = getattr(c, 'role', 'unknown')
                p_count = len(getattr(c, 'parts', []))
                debug_print(f"  Msg {i} [{role}]: {p_count} parts")

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=gemini_tools if gemini_tools else None,
            temperature=kwargs.get("temperature", 0.7),
            max_output_tokens=kwargs.get("max_tokens"),
            safety_settings=[
                types.SafetySetting(
                    category=cat,
                    threshold="BLOCK_NONE"
                ) for cat in [
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_CIVIC_INTEGRITY"
                ]
            ]
        )

        try:
            # The core Vertex AI call
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )

            if HERMES_DEBUG:
                # Log a summary of the response candidates
                c_count = len(response.candidates) if response.candidates else 0
                debug_print(f"Vertex Gemini Raw Response: {c_count} candidates")
                if response.candidates:
                    for i, cand in enumerate(response.candidates):
                        fr = getattr(cand, 'finish_reason', 'none')
                        debug_print(f"  Candidate {i} finish_reason: {fr}")
                        if cand.content and cand.content.parts:
                           debug_print(f"  Candidate {i} has {len(cand.content.parts)} parts")
                else:
                    # Log prompt feedback if candidates are empty (usually means PROMPT_BLOCKED)
                    if hasattr(response, 'prompt_feedback'):
                        debug_print(f"Prompt Feedback: {response.prompt_feedback}")

            # 1. Extraction
            content = ""
            hermes_tool_calls = []
            finish_reason = "stop"

            if response.candidates:
                candidate = response.candidates[0]
                
                # Check finish reason
                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    finish_reason = str(candidate.finish_reason).lower()
                    if HERMES_DEBUG:
                        debug_print(f"Gemini response candidate 0 finish_reason: {finish_reason}")
                    if "safety" in finish_reason:
                        content = "⚠️ [Response blocked by safety filters]"
                    elif "max_tokens" in finish_reason or "length" in finish_reason:
                        finish_reason = "length"

                if candidate.content and candidate.content.parts:
                    if HERMES_DEBUG:
                        debug_print(f"Gemini response candidate 0 has {len(candidate.content.parts)} parts")
                    for idx, part in enumerate(candidate.content.parts):
                        # Text content
                        if hasattr(part, "text") and part.text is not None:
                            if HERMES_DEBUG:
                                debug_print(f"Part {idx}: text found ({len(part.text)} chars)")
                            content += part.text

                        # Function call
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            if HERMES_DEBUG:
                                debug_print(f"Part {idx}: function_call found: {getattr(fc, 'name', 'unknown')}")
                            original_name = tool_name_map.get(
                                getattr(fc, "name", ""), getattr(fc, "name", "")
                            )
                            args = getattr(fc, "args", {}) or {}
                            hermes_tool_calls.append({
                                "id": f"call_{uuid.uuid4().hex[:12]}",
                                "type": "function",
                                "function": {
                                    "name": original_name,
                                    "arguments": json.dumps(args) if args else "{}"
                                }
                            })
                else:
                    if HERMES_DEBUG:
                        debug_print("Gemini response candidate 0 content or parts are empty")
            else:
                if HERMES_DEBUG:
                    debug_print("Gemini response has no candidates")

            # Check for empty response (Hermes needs something to act on)
            if not content and not hermes_tool_calls:
                msg = f"Gemini returned an empty response (no text or tool calls). Finish reason: {finish_reason}. Candidates: {len(response.candidates) if response.candidates else 0}"
                if HERMES_DEBUG:
                    debug_print(msg)
                logger.warning(msg)
                if response.candidates:
                    content = f"The model returned no content. (Finish reason: {finish_reason})"
                else:
                    content = "The model returned an empty response with no candidates."

            # 2. Build the Message Object (always set tool_calls — Hermes uses
            # assistant_message.tool_calls directly; OpenAI uses None when absent.)
            tool_calls_attr = (
                _vertex_tool_calls_to_openai_objects(hermes_tool_calls)
                if hermes_tool_calls
                else None
            )
            message_obj = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls_attr,
            }

            # 3. The OpenAI Envelope (satisfies Hermes' internal validation)
            openai_envelope = SimpleNamespace(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                object="chat.completion",
                created=int(time.time()),
                model=self.model_name,
                choices=[SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(**message_obj),
                    finish_reason="tool_calls" if hermes_tool_calls else finish_reason
                )],
                usage=SimpleNamespace(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0
                )
            )

            return openai_envelope

        except Exception as e:
            # Map Google GenAI errors to OpenAI standard exceptions for Hermes retries/fallbacks
            from openai import APIConnectionError, APIStatusError
            
            err_msg = str(e)
            # Always print the raw error for easier diagnosis
            print(f"⚠️ Vertex Gemini API Error ({type(e).__name__}): {err_msg}", file=sys.stderr)
            
            # Check for networking/connection issues
            if any(phrase in err_msg.lower() for phrase in ["connection", "timeout", "network", "unreachable", "dns"]):
                raise APIConnectionError(message=f"Vertex Gemini connection error: {err_msg}", request=None) from e
            
            # Handle status codes
            status_code = getattr(e, "status_code", None)
            if status_code:
                raise APIStatusError(message=err_msg, response=None, body=None) from e

            # Fallback for other errors
            raise e

    def list_models(self) -> list[str]:
        """List available models for Vertex Gemini.

        Returns a list of model IDs, or empty list if not available.
        """
        try:
            from google.genai import types

            if hasattr(self.client.models, 'list'):
                response = self.client.models.list()
                models = []
                for model in response:
                    models.append(model.name)
                return models
        except Exception as e:
            logger.debug(f"Vertex Gemini list_models error: {e}")

        return []

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in the message sequence."""
        system_instruction, contents = _format_messages_for_gemini(messages)

        response = self.client.models.count_tokens(
            model=self.model_name,
            contents=contents
        )
        return response.total_tokens


def build_vertex_gemini_client(model_name: str = "gemini-2.5-flash") -> VertexGeminiProvider:
    """Factory function to build a Vertex Gemini provider."""
    return VertexGeminiProvider(model_name=model_name)
