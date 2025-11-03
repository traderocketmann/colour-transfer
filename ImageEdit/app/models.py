"""Data models for the Color Transfer Assistant."""

import base64
import io
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image


def _encode_image_to_data_url(image_bytes: bytes, mime_type: Optional[str]) -> str:
    """Create a data URL for previewing an uploaded image in the UI."""
    guessed_mime = mime_type or 'image/png'
    encoded = base64.b64encode(image_bytes).decode('ascii')
    return f'data:{guessed_mime};base64,{encoded}'


def cleanup_thumbnail(thumbnail_path: Optional[str]) -> None:
    """Delete a thumbnail file from disk.

    Args:
        thumbnail_path: Path to the thumbnail file to delete
    """
    if thumbnail_path and os.path.exists(thumbnail_path):
        try:
            os.remove(thumbnail_path)
        except Exception:
            pass  # Silently ignore errors during cleanup


def compress_image(image_bytes: bytes, mime_type: Optional[str] = None, max_width: int = 200, quality: int = 70) -> str:
    """Compress an image to WebP format and save to disk.

    Args:
        image_bytes: Original image bytes
        mime_type: MIME type of the image (unused, kept for compatibility)
        max_width: Maximum width for the thumbnail (maintains aspect ratio)
        quality: WebP quality (1-100, lower = smaller files)

    Returns:
        Path to the saved thumbnail file
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Convert RGBA/P to RGB for better compatibility
        if img.mode in ('RGBA', 'P', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                background.paste(img, mask=img.split()[-1])
            else:
                background.paste(img)
            img = background

        # Resize if larger than max_width
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

        # Generate unique filename
        thumbnail_id = uuid.uuid4().hex
        thumbnail_path = f'/app/thumbnails/{thumbnail_id}.webp'

        # Save as WebP with aggressive compression
        img.save(thumbnail_path, format='WEBP', quality=quality, method=6, optimize=True)

        return thumbnail_path
    except Exception:
        # If compression fails, save original as-is
        thumbnail_id = uuid.uuid4().hex
        thumbnail_path = f'/app/thumbnails/{thumbnail_id}.dat'
        with open(thumbnail_path, 'wb') as f:
            f.write(image_bytes)
        return thumbnail_path


@dataclass
class ImageDescription:
    """Represents a single reusable description for an image."""

    id: str
    text: str = ''
    color_description: str = ''  # Detailed color description for reference images
    fields: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize stored fields and keep legacy fields synchronized."""
        if self.fields is None:
            self.fields = {}
        else:
            # Ensure standard dict with string keys
            self.fields = {
                str(key): (value or '').strip()
                for key, value in self.fields.items()
                if isinstance(key, str)
            }

        # Legacy key migration
        legacy_source = self.fields.pop('left_objects', None)
        legacy_reference = self.fields.pop('right_colors', None)
        legacy_reference_desc = self.fields.pop('right_color_description', None)
        if legacy_source and not self.fields.get('source_objects'):
            self.fields['source_objects'] = legacy_source
        if legacy_reference and not self.fields.get('reference_colors'):
            self.fields['reference_colors'] = legacy_reference
        if legacy_reference_desc and not self.fields.get('reference_color_description'):
            self.fields['reference_color_description'] = legacy_reference_desc

        if self.text and not self.fields.get('source_objects'):
            self.fields['source_objects'] = self.text.strip()
        elif not self.text and self.fields.get('source_objects'):
            self.text = self.fields['source_objects']

        if self.fields.get('reference_colors') and not self.text:
            self.text = self.fields['reference_colors']
        elif self.text and not self.fields.get('reference_colors'):
            # Only mirror when source field absent to preserve explicit overrides
            self.fields['reference_colors'] = self.text.strip()

        if self.color_description and not self.fields.get('reference_color_description'):
            self.fields['reference_color_description'] = self.color_description.strip()
        elif not self.color_description and self.fields.get('reference_color_description'):
            self.color_description = self.fields['reference_color_description']

    def get_field(self, name: str) -> str:
        """Return the stored value for a named variable."""
        value = self.fields.get(name, '')
        if value:
            return value
        if name == 'source_objects':
            return self.text
        if name == 'reference_colors':
            return self.text
        if name == 'reference_color_description':
            return self.color_description
        # Legacy fallbacks
        if name == 'left_objects':
            return self.fields.get('source_objects', self.text)
        if name == 'right_colors':
            return self.fields.get('reference_colors', self.text)
        if name == 'right_color_description':
            return self.fields.get('reference_color_description', self.color_description)
        return ''

    def set_field(self, name: str, value: str) -> None:
        """Set the value for a named variable and keep legacy fields in sync."""
        trimmed = value.strip()
        if trimmed:
            self.fields[name] = trimmed
        else:
            self.fields.pop(name, None)
        if name == 'source_objects' or name == 'left_objects':
            self.text = trimmed
        elif name == 'reference_colors' or name == 'right_colors':
            self.text = trimmed
        elif name == 'reference_color_description' or name == 'right_color_description':
            self.color_description = trimmed

    def all_fields(self) -> Dict[str, str]:
        """Return a dictionary of all known fields, including legacy aliases."""
        merged = dict(self.fields)
        if 'source_objects' not in merged and self.text:
            merged['source_objects'] = self.text
        if 'reference_colors' not in merged and self.text:
            merged['reference_colors'] = self.text
        if 'reference_color_description' not in merged and self.color_description:
            merged['reference_color_description'] = self.color_description
        # Retain legacy keys for backwards compatibility
        if self.text:
            merged.setdefault('left_objects', merged.get('source_objects', self.text))
            merged.setdefault('right_colors', merged.get('reference_colors', self.text))
        if self.color_description:
            merged.setdefault('right_color_description', merged.get('reference_color_description', self.color_description))
        return merged


@dataclass
class UploadedImage:
    """Holds metadata, binary content, and cached descriptions for an uploaded image."""

    id: str
    name: str
    data: bytes
    mime_type: Optional[str] = None
    thumbnail_path: Optional[str] = None  # Path to thumbnail on server
    full_image_path: Optional[str] = None  # Path to full image on server
    _data_url: Optional[str] = field(default=None, init=False, repr=False)
    descriptions: List[ImageDescription] = field(default_factory=list)
    selected_description_id: Optional[str] = None

    @property
    def data_url(self) -> str:
        """Lazy generation of data URL - only encode when needed (full resolution)."""
        if self._data_url is None:
            self._data_url = _encode_image_to_data_url(self.data, self.mime_type)
        return self._data_url

    @property
    def thumbnail_url(self) -> str:
        """Return URL path for serving thumbnail from server."""
        if self.thumbnail_path:
            # Extract just the filename from the path
            filename = self.thumbnail_path.split('/')[-1]
            return f'/thumbnails/{filename}'
        # Fallback to full image data URL if no thumbnail
        return self.data_url


def serialize_description(description: ImageDescription) -> Dict[str, Any]:
    """Serialize a description to a dict for storage."""
    return {
        'id': description.id,
        'text': description.text,
        'color_description': description.color_description,
        'fields': description.all_fields(),
    }


def deserialize_description(payload: Dict[str, Any]) -> ImageDescription:
    """Deserialize a description from storage."""
    return ImageDescription(
        id=payload.get('id') or uuid.uuid4().hex,
        text=payload.get('text', ''),
        color_description=payload.get('color_description', ''),
        fields=payload.get('fields') or {},
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
        descriptions = [ImageDescription(id=default_id)]
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


def get_description_by_id(image: UploadedImage, description_id: Optional[str]) -> Optional[ImageDescription]:
    """Get a description by its ID."""
    if description_id is None:
        return None
    for description in image.descriptions:
        if description.id == description_id:
            return description
    return None


def get_selected_description(image: UploadedImage) -> Optional[ImageDescription]:
    """Get the currently selected description for an image."""
    desc = get_description_by_id(image, image.selected_description_id)
    if desc is None and image.descriptions:
        desc = image.descriptions[0]
        image.selected_description_id = desc.id
    return desc


def get_selected_description_text(image: UploadedImage) -> str:
    """Get the text of the currently selected description."""
    description = get_selected_description(image)
    if description is None:
        return ''
    return description.text.strip()


def set_selected_description(image: UploadedImage, description_id: str) -> None:
    """Set the selected description by ID."""
    if get_description_by_id(image, description_id) is not None:
        image.selected_description_id = description_id


def update_description_text(
    image: UploadedImage,
    description_id: Optional[str],
    text: str,
    *,
    variable_name: Optional[str] = None,
) -> None:
    """Update the text of a description."""
    description = get_description_by_id(image, description_id)
    target_name = (variable_name or 'source_objects')
    trimmed = text.strip()
    if description is None:
        if not image.descriptions:
            new_description = ImageDescription(id=uuid.uuid4().hex)
            new_description.set_field(target_name, trimmed)
            image.descriptions.append(new_description)
            image.selected_description_id = new_description.id
        return
    description.set_field(target_name, trimmed)


def add_description_entry(image: UploadedImage, text: str = '') -> ImageDescription:
    """Add a new description entry to an image."""
    new_description = ImageDescription(id=uuid.uuid4().hex)
    if text:
        new_description.set_field('source_objects', text)
    image.descriptions.append(new_description)
    image.selected_description_id = new_description.id
    return new_description


def remove_description_entry(image: UploadedImage, description_id: Optional[str]) -> bool:
    """Remove a description entry from an image."""
    if description_id is None:
        return False
    remaining: List[ImageDescription] = []
    removed = False
    for description in image.descriptions:
        if description.id == description_id:
            removed = True
            continue
        remaining.append(description)
    if removed:
        image.descriptions = remaining
        # Set selected_description_id to first remaining description or None if empty
        image.selected_description_id = image.descriptions[0].id if image.descriptions else None
    return removed


def get_variable_value(image: UploadedImage, variable_name: str) -> str:
    """Return the stored value for a named variable on the selected description."""
    description = get_selected_description(image)
    if description is None:
        return ''
    return description.get_field(variable_name).strip()


def set_variable_value(image: UploadedImage, variable_name: str, value: str) -> None:
    """Set the value for a named variable on the selected description."""
    description = get_selected_description(image)
    if description is None:
        description = add_description_entry(image)
    description.set_field(variable_name, value)


def is_variable_filled(image: UploadedImage, variable_name: str) -> bool:
    """Return True when a variable has a non-empty value."""
    return bool(get_variable_value(image, variable_name))


def get_all_variable_values(image: UploadedImage) -> Dict[str, str]:
    """Return all stored variable values for the selected description."""
    description = get_selected_description(image)
    if description is None:
        return {}
    return description.all_fields()
