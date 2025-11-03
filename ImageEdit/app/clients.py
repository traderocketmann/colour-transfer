"""API clients for external services."""

import asyncio
import base64
import json
import os
import uuid
from typing import Any, Callable, Dict, List, Optional

import httpx
import websockets
from openai import OpenAI

from models import UploadedImage, _encode_image_to_data_url


# Configuration
VISION_MODEL = os.getenv('VISION_MODEL') or 'gpt-5-mini'
COMFYUI_BASE_URL = os.getenv('COMFYUI_BASE_URL') or 'http://host.docker.internal:8130'


class WorkflowError(RuntimeError):
    """Raised when executing a workflow fails."""


def extract_output_text(response: Any) -> str:
    """Normalize text extraction from Responses API replies."""
    # New OpenAI SDKs expose output_text directly for convenience.
    if hasattr(response, 'output_text') and response.output_text:
        return response.output_text.strip()

    # Fallback to the raw dict representation.
    try:
        data = response.model_dump()
    except AttributeError:
        try:
            data = json.loads(response.model_dump_json())
        except Exception:
            data = response

    outputs = data.get('output', [])
    for item in outputs:
        for content in item.get('content', []):
            if content.get('type') in ('output_text', 'text'):
                text = content.get('text')
                if text:
                    return text.strip()

    # A final attempt if the structure differs.
    if isinstance(data, dict):
        message = data.get('output_text') or data.get('text')
        if message:
            return str(message).strip()

    return ''


def get_usage_and_cost(response: Any) -> Optional[Dict[str, Any]]:
    """Extract token usage and calculate approximate cost for a response."""
    usage = getattr(response, 'usage', None)
    if usage is None:
        return None

    input_tokens = getattr(usage, 'input_tokens', 0) or 0
    output_tokens = getattr(usage, 'output_tokens', 0) or 0
    total_tokens = getattr(usage, 'total_tokens', input_tokens + output_tokens) or 0

    input_details = getattr(usage, 'input_tokens_details', None)
    cached_tokens = getattr(input_details, 'cached_tokens', 0) if input_details else 0

    output_details = getattr(usage, 'output_tokens_details', None)
    reasoning_tokens = getattr(output_details, 'reasoning_tokens', 0) if output_details else 0

    cached_input_cost = (cached_tokens / 1_000_000) * 0.005
    net_input_tokens = max(input_tokens - cached_tokens, 0)
    input_cost = (net_input_tokens / 1_000_000) * 0.05
    output_cost = (output_tokens / 1_000_000) * 0.4

    total_cost = input_cost + output_cost + cached_input_cost

    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'cached_tokens': cached_tokens,
        'reasoning_tokens': reasoning_tokens,
        'cost': total_cost,
    }


class VisionClient:
    """Thin wrapper around the OpenAI Responses API for image understanding."""

    def __init__(self, *, model: str = VISION_MODEL, api_key: Optional[str] = None) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key)

    async def generate_variables(
        self,
        source_image: UploadedImage,
        reference_image: UploadedImage,
        *,
        left_requests: Optional[List[Dict[str, Any]]] = None,
        right_requests: Optional[List[Dict[str, Any]]] = None,
        known_left: Optional[Dict[str, str]] = None,
        known_right: Optional[Dict[str, str]] = None,
    ) -> tuple[Dict[str, str], Optional[Dict[str, Any]]]:
        """Generate requested variable values for a source/reference image pair."""
        requests_left = list(left_requests or [])
        requests_right = list(right_requests or [])
        if not requests_left and not requests_right:
            return {}, None

        left_url = source_image.data_url or _encode_image_to_data_url(source_image.data, source_image.mime_type)
        right_url = reference_image.data_url or _encode_image_to_data_url(reference_image.data, reference_image.mime_type)

        requested_fields = [req.get('name') for req in (*requests_left, *requests_right)]
        requested_fields = [name for name in requested_fields if name]

        instruction_lines: List[str] = [
            'You will examine a source image (objects to recolor) and a reference image (desired palette).',
        ]
        if requested_fields:
            joined_fields = ', '.join(f'"{name}"' for name in requested_fields)
            instruction_lines.append(f'Return a JSON object containing ONLY the following string fields: {joined_fields}.')
        if known_left:
            instruction_lines.append(f'Previously filled source variables: {json.dumps(known_left, ensure_ascii=False)}.')
        if known_right:
            instruction_lines.append(f'Previously filled reference variables: {json.dumps(known_right, ensure_ascii=False)}.')
        if requests_left or requests_right:
            instruction_lines.append('Follow these guidelines for each requested field:')
            for req in requests_left + requests_right:
                name = req.get('name')
                guidance = req.get('instruction') or f'Provide a concise description for "{name}".'
                instruction_lines.append(f'- "{name}": {guidance}')
        instruction_lines.append('Respond with valid JSON only. Do not include narrative text.')
        instruction = ' '.join(instruction_lines)

        def _call() -> Any:
            return self._client.responses.create(
                model=self.model,
                text={'format': {'type': 'json_object'}},
                input=[
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'input_text', 'text': instruction},
                            {'type': 'input_text', 'text': 'Source image (to recolor):'},
                            {'type': 'input_image', 'image_url': left_url},
                            {'type': 'input_text', 'text': 'Reference image (desired palette):'},
                            {'type': 'input_image', 'image_url': right_url},
                        ],
                    }
                ],
            )

        response = await asyncio.to_thread(_call)
        usage_data = get_usage_and_cost(response)
        raw_text = extract_output_text(response)
        try:
            data = json.loads(raw_text) if raw_text else {}
        except Exception:
            data = {}

        if requested_fields:
            allowed = set(requested_fields)
            filtered = {}
            for key, value in (data or {}).items():
                if key in allowed and isinstance(value, str) and value.strip():
                    filtered[key] = value.strip()
            data = filtered
        return data, usage_data


class ComfyUIClient:
    """Helper to upload assets and trigger workflows via the ComfyUI API."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        configured_base_url = base_url or COMFYUI_BASE_URL
        self.base_url = configured_base_url.rstrip('/')
        ws_base = self.base_url.replace('http', 'ws', 1)
        self.ws_endpoint = f'{ws_base}/ws'
        self._http_client: Optional[httpx.AsyncClient] = None
        self._active_prompt_id: Optional[str] = None

    async def _ensure_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        return self._http_client

    async def close(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def upload_image(self, image: UploadedImage) -> str:
        """Upload an image to the ComfyUI input directory."""
        client = await self._ensure_http_client()
        filename = image.name or f'upload-{uuid.uuid4().hex}.png'
        files = {'image': (filename, image.data, image.mime_type or 'image/png')}
        response = await client.post('/upload/image', files=files)
        response.raise_for_status()
        payload = response.json()
        return payload.get('name', filename)

    async def upload_image_bytes(self, image_bytes: bytes, filename: Optional[str] = None, mime_type: Optional[str] = None) -> str:
        """Upload raw image bytes to the ComfyUI input directory."""
        client = await self._ensure_http_client()
        upload_filename = filename or f'upload-{uuid.uuid4().hex}.png'
        files = {'image': (upload_filename, image_bytes, mime_type or 'image/png')}
        response = await client.post('/upload/image', files=files)
        response.raise_for_status()
        payload = response.json()
        return payload.get('name', upload_filename)

    async def run_workflow(
        self,
        workflow: Dict[str, Any],
        *,
        progress_callback: Optional[Callable[[float, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Queue a workflow and wait for completion via WebSocket notifications."""
        client_id = uuid.uuid4().hex
        queue_payload = {'prompt': workflow, 'client_id': client_id}

        client = await self._ensure_http_client()
        post_response = await client.post('/prompt', json=queue_payload)
        post_response.raise_for_status()
        data = post_response.json()
        prompt_id = data.get('prompt_id')
        if prompt_id is None:
            raise WorkflowError('ComfyUI did not return a prompt_id.')
        self._active_prompt_id = prompt_id

        ws_url = f'{self.ws_endpoint}?clientId={client_id}'
        completed = False
        current_node: Optional[str] = None
        output_images: Dict[str, List[bytes]] = {}
        try:
            async with websockets.connect(ws_url, max_size=None) as websocket:
                async for message in websocket:
                    if isinstance(message, bytes):
                        if current_node is not None:
                            # Skip the first 8 header bytes (per ComfyUI websocket protocol).
                            payload_bytes = message[8:]
                            if payload_bytes:
                                output_images.setdefault(current_node, []).append(payload_bytes)
                        continue

                    payload = json.loads(message)
                    message_type = payload.get('type')
                    data = payload.get('data', {})

                    if message_type == 'status':
                        if data.get('prompt_id') != prompt_id:
                            continue
                        status_info = data.get('status', {})
                        status_value = status_info.get('status')
                        if status_value in {'completed', 'success'} or status_info.get('completed'):
                            completed = True
                            break
                        if status_value == 'error':
                            error_message = status_info.get('error', 'Unknown error')
                            # Log full status data for debugging
                            print(f"[ComfyUI] Status error event received: {json.dumps(data, indent=2)}")
                            raise WorkflowError(error_message)
                        continue

                    if message_type == 'execution_success' and data.get('prompt_id') == prompt_id:
                        completed = True
                        break
                    if message_type == 'execution_success':
                        continue

                    if message_type == 'executing':
                        if data.get('prompt_id') != prompt_id:
                            continue
                        node_id = data.get('node')
                        current_node = node_id
                        if node_id is None:
                            completed = True
                            break
                        continue

                    if message_type == 'progress':
                        if data.get('prompt_id') != prompt_id:
                            continue
                        current_node = data.get('node') or current_node
                        value = data.get('value')
                        max_value = data.get('max')
                        if (
                            progress_callback is not None
                            and value is not None
                            and max_value not in (None, 0)
                        ):
                            fraction = max(0.0, min(float(value) / float(max_value), 1.0))
                            progress_callback(fraction, data)
                        continue
                    if message_type == 'progress_state':
                        continue
                    if message_type == 'execution_start':
                        continue
                    if message_type == 'execution_error':
                        error = data.get('error', 'Unknown error')
                        # Log full event data for debugging
                        print(f"[ComfyUI] Execution error event received: {json.dumps(data, indent=2)}")
                        raise WorkflowError(error)
                    if message_type == 'executed' and data.get('prompt_id') == prompt_id:
                        continue

                    # Log unhandled message types for debugging
                    if message_type not in {'status', 'execution_success', 'executing', 'progress', 'execution_start', 'execution_error', 'executed'}:
                        print(f"[ComfyUI] Unhandled message type '{message_type}': {json.dumps(payload, indent=2)}")

            if not completed:
                raise WorkflowError('Workflow did not complete before the connection closed.')

            history_response = await client.get(f'/history/{prompt_id}')
            history_response.raise_for_status()
            history_data = history_response.json()

            websocket_images = []
            for node_id, node_images in output_images.items():
                for idx, image_bytes in enumerate(node_images, start=1):
                    base64_data = base64.b64encode(image_bytes).decode('ascii')
                    websocket_images.append(
                        {
                            'bytes': image_bytes,
                            'data_url': f'data:image/png;base64,{base64_data}',
                            'mime_type': 'image/png',
                            'filename': f'{prompt_id}-{node_id}-{idx}.png',
                            'node_id': node_id,
                        }
                    )
            return {
                'history': history_data,
                'websocket_images': websocket_images,
            }
        finally:
            self._active_prompt_id = None

    async def interrupt(self) -> None:
        """Send an interrupt signal to stop the current workflow."""
        client = await self._ensure_http_client()
        payload: Dict[str, Any] = {}
        if self._active_prompt_id:
            payload['prompt_id'] = self._active_prompt_id
        try:
            if payload:
                await client.post('/interrupt', json=payload)
            else:
                await client.post('/interrupt')
        except httpx.HTTPStatusError as error:
            raise WorkflowError(f'Interrupt request failed: {error.response.text}') from error
        except Exception as error:
            raise WorkflowError(f'Failed to send interrupt request: {error}') from error
