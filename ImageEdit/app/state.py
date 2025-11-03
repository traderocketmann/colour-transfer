"""State management for the Color Transfer Assistant."""

import asyncio
import base64
import binascii
import os
import uuid
from typing import Any, Callable, Dict, List, Optional

from nicegui import app, ui

from models import ImageDescription, UploadedImage, is_variable_filled
from prompt_settings import (
    DEFAULT_VARIABLE_DEFINITIONS,
    normalize_definitions,
    variables_by_role,
    extract_variables_from_blocks,
    build_definition_lookup,
)


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError as error:
        raise ValueError(f'{name} must be an integer, got {value!r}') from error


DEFAULT_BATCH_SIZE = _get_int_env('DEFAULT_BATCH_SIZE', 4)
PERSISTENCE_ENABLED = os.getenv('ENABLE_CLIENT_STORAGE', 'true').strip().lower() in {'1', 'true', 'yes'}
DOWNLOADS_DIR = '/app/downloads'
os.makedirs(DOWNLOADS_DIR, exist_ok=True)


def serialize_description(description: ImageDescription) -> Dict[str, Any]:
    """Serialize a description to a dict for storage."""
    return {
        'id': description.id,
        'text': description.text,
        'color_description': description.color_description,
    }


def deserialize_description(payload: Dict[str, Any]) -> ImageDescription:
    """Deserialize a description from storage."""
    return ImageDescription(
        id=payload.get('id') or uuid.uuid4().hex,
        text=payload.get('text', ''),
        color_description=payload.get('color_description', ''),
    )


def serialize_image(image: UploadedImage) -> Dict[str, Any]:
    """Serialize an image to a dict for storage.

    NOTE: Both thumbnail and full image paths are stored on server.
    Images persist across browser refreshes.
    """
    return {
        'id': image.id,
        'name': image.name,
        'mime_type': image.mime_type,
        'thumbnail_path': image.thumbnail_path,
        'full_image_path': image.full_image_path,
        'descriptions': [serialize_description(desc) for desc in image.descriptions],
        'selected_description_id': image.selected_description_id,
    }


def deserialize_image(payload: Dict[str, Any]) -> UploadedImage:
    """Deserialize an image from storage.

    NOTE: Full images are loaded from server disk if available.
    Thumbnails and metadata are loaded from browser storage.
    """
    descriptions_payload = payload.get('descriptions', []) or []
    descriptions = [deserialize_description(item) for item in descriptions_payload]
    if not descriptions:
        default_id = uuid.uuid4().hex
        descriptions = [ImageDescription(id=default_id, text='')]
        selected_description_id = default_id
    else:
        selected_description_id = payload.get('selected_description_id') or descriptions[0].id
        # Validate that selected_description_id exists in descriptions
        valid_ids = {desc.id for desc in descriptions}
        if selected_description_id not in valid_ids:
            selected_description_id = descriptions[0].id

    # Load full image data from server if path exists
    full_image_path = payload.get('full_image_path')
    image_data = b''
    if full_image_path and os.path.exists(full_image_path):
        try:
            with open(full_image_path, 'rb') as f:
                image_data = f.read()
        except Exception:
            pass  # If file doesn't exist or can't be read, data remains empty

    image = UploadedImage(
        id=payload.get('id') or uuid.uuid4().hex,
        name=payload.get('name', 'Image'),
        data=image_data,
        mime_type=payload.get('mime_type'),
        thumbnail_path=payload.get('thumbnail_path'),
        full_image_path=full_image_path,
        descriptions=descriptions,
        selected_description_id=selected_description_id,
    )

    return image


def _decode_data_url(data_url: Optional[str]) -> tuple[Optional[bytes], Optional[str]]:
    """Decode a data URL into raw bytes and mime type."""
    if not data_url or ',' not in data_url:
        return None, None
    header, encoded = data_url.split(',', 1)
    mime_type = None
    if header.startswith('data:'):
        meta = header[5:]
        if ';' in meta:
            mime_type = meta.split(';', 1)[0]
        else:
            mime_type = meta or None
    try:
        return base64.b64decode(encoded), mime_type
    except (ValueError, binascii.Error):
        return None, mime_type


class ProcessingState:
    """Maintain UI state for uploaded assets and controls."""

    def __init__(self) -> None:
        self.source_images: List[UploadedImage] = []
        self.reference_images: List[UploadedImage] = []
        self.prompt_blocks: Optional[List[Dict[str, str]]] = None
        self.variable_definitions: Optional[List[Dict[str, Any]]] = None
        self.prompt_builder_refresher: Optional[Callable[[], None]] = None
        self.prompt_enhancer_client: Optional[Any] = None
        self.prompt_template_text: str = ''
        self.vision_client: Optional[Any] = None
        self.comfy_client: Optional[Any] = None
        self.workflow_progress: Optional[Any] = None
        self.workflow_progress_label: Optional[Any] = None
        self.overall_progress: Optional[Any] = None
        self.log: Optional[ui.log] = None
        self.log_history: List[str] = []
        self.output_images: List[Dict[str, Any]] = []
        self.output_container: Optional[ui.column] = None
        self.source_container: Optional[ui.column] = None
        self.reference_container: Optional[ui.column] = None
        self.generate_missing_button: Optional[Any] = None
        self.one_click_button: Optional[ui.button] = None
        self.stop_button: Optional[ui.button] = None
        self.storage_loaded = False
        self.processing_task: Optional[asyncio.Task] = None
        self.stop_requested: bool = False
        self.persistence_enabled = PERSISTENCE_ENABLED
        self.session_id: Optional[str] = None
        # Progress state (persisted values to restore on reconnect)
        self.overall_progress_value: float = 0.0
        self.workflow_progress_value: float = 0.0
        self.workflow_progress_text: str = 'ComfyUI progress will appear here.'

    def set_log(self, log_widget: ui.log) -> None:
        """Set the log widget for displaying messages."""
        self.log = log_widget
        for message in self.log_history:
            self.log.push(message)

    def log_message(self, message: str) -> None:
        """Log a message to the UI log widget."""
        self.log_history.append(message)
        if len(self.log_history) > 200:
            self.log_history = self.log_history[-200:]
        if self.log is not None:
            self.log.push(message)

    def set_output_container(self, container: ui.column) -> None:
        """Set the container for output images."""
        self.output_container = container

    def ensure_variable_definitions(self) -> List[Dict[str, Any]]:
        """Ensure variable definitions are initialized and normalized."""
        if self.variable_definitions is None:
            # Always reload defaults; variable definitions are not user-editable.
            self.variable_definitions = normalize_definitions(DEFAULT_VARIABLE_DEFINITIONS)
        return self.variable_definitions

    def set_variable_definitions(self, definitions: List[Dict[str, Any]]) -> None:
        """Replace variable definitions with a normalized list and persist."""
        self.variable_definitions = normalize_definitions(definitions)
        self.persist()

    def get_active_variables_by_role(self) -> Dict[str, List[str]]:
        """Return the variables referenced in the prompt builder grouped by role."""
        definitions = self.ensure_variable_definitions()
        grouped: Dict[str, List[str]] = {'source': [], 'reference': []}
        for entry in definitions:
            role = entry.get('role')
            name = entry.get('name')
            if role in grouped and name and name not in grouped[role]:
                grouped[role].append(name)

        if not self.prompt_blocks:
            return grouped

        active = set(extract_variables_from_blocks(self.prompt_blocks))
        if not active:
            return grouped

        filtered: Dict[str, List[str]] = {'source': [], 'reference': []}
        for role, names in grouped.items():
            filtered[role] = [name for name in names if name in active]
        return filtered

    def get_variables_for_role(self, role: str) -> List[str]:
        """Return the list of variable names associated with a role in the prompt."""
        return self.get_active_variables_by_role().get(role, [])

    def register_image_container(self, role: str, container: ui.column) -> None:
        """Register a container for image cards on the specified role."""
        if role == 'source':
            self.source_container = container
        else:
            self.reference_container = container

    def render_image_column(self, role: str) -> None:
        """Render the image column for the specified role."""
        if role == 'source':
            container = self.source_container
        else:
            container = self.reference_container
        if container is None:
            return
        # Import here to avoid circular dependency
        from ui_components import render_image_column
        render_image_column(role, container, state=self)

    def render_all_images(self) -> None:
        """Render all image columns."""
        self.render_image_column('source')
        self.render_image_column('reference')

    def clear_outputs(self) -> None:
        """Clear all output images."""
        from models import cleanup_thumbnail

        # Clean up all thumbnail files
        for entry in self.output_images:
            cleanup_thumbnail(entry.get('thumbnail_path'))
            full_image_path = entry.get('full_image_path')
            if full_image_path and os.path.exists(full_image_path):
                try:
                    os.remove(full_image_path)
                except Exception:
                    pass

        self.output_images.clear()
        if self.output_container is not None:
            from ui_components import render_output_gallery
            render_output_gallery(self.output_container, self.output_images, self)

    def clear_source_images(self) -> None:
        """Clear all source images."""
        from models import cleanup_thumbnail

        # Clean up all thumbnail and full image files
        for image in self.source_images:
            cleanup_thumbnail(image.thumbnail_path)
            if image.full_image_path and os.path.exists(image.full_image_path):
                try:
                    os.remove(image.full_image_path)
                except Exception:
                    pass

        self.source_images.clear()
        self.render_image_column('source')
        self.persist()

    def clear_reference_images(self) -> None:
        """Clear all reference images."""
        from models import cleanup_thumbnail

        # Clean up all thumbnail and full image files
        for image in self.reference_images:
            cleanup_thumbnail(image.thumbnail_path)
            if image.full_image_path and os.path.exists(image.full_image_path):
                try:
                    os.remove(image.full_image_path)
                except Exception:
                    pass

        self.reference_images.clear()
        self.render_image_column('reference')
        self.persist()

    def add_output_image(self, entry: Dict[str, Any]) -> None:
        """Add an output image to the gallery."""
        # Decode data URL if raw bytes are not provided
        data_url = entry.get('data_url')
        image_bytes = entry.pop('bytes', None)
        if image_bytes is None and data_url:
            image_bytes, inferred_mime = _decode_data_url(data_url)
            if inferred_mime and not entry.get('mime_type'):
                entry['mime_type'] = inferred_mime

        # Generate thumbnail for output if we have bytes and no thumbnail yet
        if image_bytes is not None and 'thumbnail_path' not in entry:
            from models import compress_image
            mime_type = entry.get('mime_type', 'image/png')
            entry['thumbnail_path'] = compress_image(image_bytes, mime_type)

        # Persist full image to disk when bytes are available
        if image_bytes is not None:
            filename = entry.get('filename') or f'{uuid.uuid4().hex}.png'
            if '.' in filename:
                file_ext = filename.rsplit('.', 1)[-1]
            else:
                mime_type = entry.get('mime_type', 'image/png')
                file_ext = mime_type.split('/')[-1] if '/' in mime_type else 'png'
            image_id = uuid.uuid4().hex
            full_image_path = f'{DOWNLOADS_DIR}/{image_id}.{file_ext}'
            with open(full_image_path, 'wb') as file_handle:
                file_handle.write(image_bytes)
            entry['full_image_path'] = full_image_path
            # Remove embedded data URLs once persisted to disk
            entry.pop('data_url', None)
        else:
            # Ensure data_url remains available for UI fallback
            if data_url is not None:
                entry['data_url'] = data_url
            entry.setdefault('thumbnail_path', None)

        replacement_index = None
        for idx, existing in enumerate(self.output_images):
            same_pair = (
                existing.get('source_index') == entry.get('source_index')
                and existing.get('reference_index') == entry.get('reference_index')
            )
            same_label = existing.get('pair_index') == entry.get('pair_index') or (existing.get('pair_index') is None and entry.get('pair_index') is None)
            if same_pair and same_label:
                replacement_index = idx
                break

        replaced = False
        if replacement_index is not None:
            self.output_images[replacement_index] = entry
            replaced = True
        else:
            for idx, existing in enumerate(self.output_images):
                if (
                    existing.get('source_index') == entry.get('source_index')
                    and existing.get('reference_index') == entry.get('reference_index')
                ):
                    self.output_images[idx] = entry
                    replaced = True
                    break
            if not replaced:
                self.output_images.append(entry)
        if self.output_container is not None:
            from ui_components import render_output_gallery
            render_output_gallery(self.output_container, self.output_images, self)

    def _serialize_outputs(self) -> List[Dict[str, Any]]:
        """Serialize output metadata for storage.

        NOTE: Output entries store server-side paths for thumbnails and full images.
        Binary data itself remains on disk.
        """
        serialized = []
        for entry in self.output_images:
            # Get thumbnail path (server-side file path)
            thumbnail_path = entry.get('thumbnail_path')

            serialized.append(
                {
                    'pair_index': entry.get('pair_index'),
                    'filename': entry.get('filename'),
                    'thumbnail_path': thumbnail_path,
                    'mime_type': entry.get('mime_type'),
                    'pair_number': entry.get('pair_number'),
                    'source_description': entry.get('source_description'),
                    'reference_description': entry.get('reference_description'),
                    'source_index': entry.get('source_index'),
                    'reference_index': entry.get('reference_index'),
                    'full_image_path': entry.get('full_image_path'),
                }
            )
        return serialized

    def _load_outputs_from_payload(self, payload: List[Dict[str, Any]]) -> None:
        """Rehydrate output entries from stored metadata.

        NOTE: Only server-side paths and metadata are restored; image files stay on disk.
        """
        self.output_images = []
        for item in payload or []:
            self.output_images.append(
                {
                    'pair_index': item.get('pair_index'),
                    'filename': item.get('filename'),
                    'thumbnail_path': item.get('thumbnail_path'),
                    'mime_type': item.get('mime_type'),
                    'pair_number': item.get('pair_number'),
                    'source_description': item.get('source_description'),
                    'reference_description': item.get('reference_description'),
                    'source_index': item.get('source_index'),
                    'reference_index': item.get('reference_index'),
                    'bytes': None,
                    'full_image_path': item.get('full_image_path'),
                }
            )

    def load_from_storage(self) -> None:
        """Load state from browser storage."""
        if self.storage_loaded:
            return
        if not self.persistence_enabled:
            self.storage_loaded = True
            return
        if app.storage.secret is None:
            return
        payload = app.storage.user.get('images')
        if not payload:
            self.storage_loaded = True
            return
        source_payload = payload.get('source')
        reference_payload = payload.get('reference')
        if source_payload is None and 'left' in payload:
            source_payload = payload.get('left', [])
        if reference_payload is None and 'right' in payload:
            reference_payload = payload.get('right', [])
        self.source_images = [deserialize_image(item) for item in (source_payload or [])]
        self.reference_images = [deserialize_image(item) for item in (reference_payload or [])]
        # Note: batch_size removed but kept in storage for backwards compatibility
        self.prompt_blocks = payload.get('prompt_blocks')
        if self.prompt_blocks:
            migrated_blocks: List[Dict[str, Any]] = []
            allowed_variables = {item['name'] for item in DEFAULT_VARIABLE_DEFINITIONS}
            for block in self.prompt_blocks:
                if block.get('type') == 'variable':
                    name = block.get('name')
                    if name == 'left_objects':
                        block['name'] = 'source_objects'
                    elif name == 'right_colors':
                        block['name'] = 'reference_colors'
                    elif name in {'right_color_description', 'reference_color_description'}:
                        block = {
                            'id': block.get('id') or uuid.uuid4().hex,
                            'type': 'text',
                            'content': '',
                        }
                    if block.get('type') == 'variable' and block.get('name') not in allowed_variables:
                        continue
                migrated_blocks.append(block)
            self.prompt_blocks = migrated_blocks or []
        # Variable definitions are not user editable; always use defaults.
        self.variable_definitions = None
        self.prompt_template_text = payload.get('prompt_template') or ''
        self._load_outputs_from_payload(payload.get('outputs', []))
        self.storage_loaded = True

    def persist(self) -> None:
        """Persist state to browser storage."""
        if self.persistence_enabled and not self.is_processing():
            try:
                storage = app.storage.user
            except RuntimeError:
                return
            if app.storage.secret is None:
                return
            storage['images'] = {
                'source': [serialize_image(item) for item in self.source_images],
                'reference': [serialize_image(item) for item in self.reference_images],
                'prompt_blocks': self.prompt_blocks,
                'prompt_template': self._current_prompt_template(),
                'outputs': self._serialize_outputs(),
            }
        # Update button visibility after state changes
        self.update_missing_button_visibility()

    def _current_prompt_template(self) -> str:
        """Return the current prompt template string for persistence."""
        if not self.prompt_blocks:
            return self.prompt_template_text or ''
        try:
            from prompt_builder import blocks_to_template  # Local import to avoid circular dependency

            template = blocks_to_template(self.prompt_blocks)
        except Exception:
            template = self.prompt_template_text or ''
        self.prompt_template_text = template
        return template

    def has_missing_descriptions(self) -> bool:
        """Check if there are any images with missing descriptions."""
        required = self.get_active_variables_by_role()
        required_source = required.get('source', [])
        required_reference = required.get('reference', [])
        for img in self.source_images:
            for variable in required_source:
                if not is_variable_filled(img, variable):
                    return True
        for img in self.reference_images:
            for variable in required_reference:
                if not is_variable_filled(img, variable):
                    return True
        return False

    def update_missing_button_visibility(self) -> None:
        """Update the visibility of the 'Generate missing descriptions' button."""
        if self.generate_missing_button is not None:
            if self.has_missing_descriptions():
                self.generate_missing_button.set_visibility(True)
            else:
                self.generate_missing_button.set_visibility(False)

    def reset_ui_elements(self) -> None:
        """Clear references to UI elements so they can be recreated for a new client view."""
        self.workflow_progress = None
        self.workflow_progress_label = None
        self.overall_progress = None
        self.log = None
        self.output_container = None
        self.source_container = None
        self.reference_container = None
        self.generate_missing_button = None
        self.one_click_button = None
        self.stop_button = None
        self.prompt_builder_refresher = None

    def ensure_overall_progress(self):
        """Ensure there is a reusable overall progress bar."""
        if self.overall_progress is None:
            progress = ui.linear_progress()
            progress.props('instant-feedback rounded')
            progress.classes('w-full')
            progress.set_value(self.overall_progress_value)
            progress.set_visibility(False)
            self.overall_progress = progress
        return self.overall_progress

    def update_overall_progress(self, value: float) -> None:
        """Update overall progress and persist the value."""
        self.overall_progress_value = value
        if self.overall_progress is not None:
            try:
                self.overall_progress.set_value(value)
            except RuntimeError:
                # Client has been deleted (e.g., user navigated away)
                pass

    def ensure_workflow_progress(self):
        """Ensure there is a reusable workflow progress bar."""
        if self.workflow_progress is None:
            progress = ui.linear_progress()
            progress.props('instant-feedback rounded')
            progress.classes('w-full')
            progress.set_value(self.workflow_progress_value)
            progress.set_visibility(False)
            self.workflow_progress = progress
        return self.workflow_progress

    def update_workflow_progress(self, value: float) -> None:
        """Update workflow progress and persist the value."""
        self.workflow_progress_value = value
        if self.workflow_progress is not None:
            try:
                self.workflow_progress.set_value(value)
            except RuntimeError:
                # Client has been deleted (e.g., user navigated away)
                pass

    def ensure_workflow_progress_label(self):
        """Ensure there is a label for workflow progress updates."""
        if self.workflow_progress_label is None:
            label = ui.label(self.workflow_progress_text)
            label.classes('text-sm text-secondary')
            label.set_visibility(False)
            self.workflow_progress_label = label
        return self.workflow_progress_label

    def update_workflow_progress_text(self, text: str) -> None:
        """Update workflow progress label text and persist it."""
        self.workflow_progress_text = text
        if self.workflow_progress_label is not None:
            try:
                self.workflow_progress_label.set_text(text)
            except RuntimeError:
                # Client has been deleted (e.g., user navigated away)
                pass

    def register_processing_controls(self, start_button: Any, stop_button: Any) -> None:
        """Register buttons used to control processing."""
        self.one_click_button = start_button
        self.stop_button = stop_button
        self.update_processing_controls()

    def is_processing(self) -> bool:
        """Return whether a processing task is currently active."""
        return self.processing_task is not None and not self.processing_task.done()

    def update_processing_controls(self) -> None:
        """Swap visibility of start/stop buttons based on processing state."""
        processing = self.is_processing()
        if self.one_click_button is not None:
            # Show one-click button when NOT processing
            self.one_click_button.set_visibility(not processing)
        if self.stop_button is not None:
            # Show stop button when processing
            self.stop_button.set_visibility(processing)

    def begin_processing(self, task: asyncio.Task) -> None:
        """Record the currently running processing task."""
        self.processing_task = task
        self.stop_requested = False
        self.update_processing_controls()

    def finish_processing(self) -> None:
        """Clear processing task references and reset controls."""
        self.processing_task = None
        self.stop_requested = False
        self.update_processing_controls()
        self.persist()

    def request_stop(self) -> None:
        """Signal that processing should stop and cancel the running task."""
        if not self.is_processing():
            return
        self.stop_requested = True
        self.log_message('Stop requested; attempting to cancel current job.')
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send_interrupt())
        except RuntimeError:
            # Not running inside an event loop; ignore.
            pass
        task = self.processing_task
        if task is not None and not task.done():
            task.cancel()

    async def _send_interrupt(self) -> None:
        """Send an interrupt request to the ComfyUI server."""
        if self.comfy_client is None:
            return
        try:
            await self.comfy_client.interrupt()
            self.log_message('Sent interrupt request to ComfyUI.')
        except Exception as error:
            self.log_message(f'Failed to interrupt ComfyUI: {error}')
