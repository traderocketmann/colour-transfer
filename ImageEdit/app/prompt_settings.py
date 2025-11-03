"""Shared utilities and defaults for prompt builder variables."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_VARIABLE_DEFINITIONS: List[Dict[str, Any]] = [
    {
        'name': 'source_objects',
        'role': 'source',
        'label': 'Source objects',
        'placeholder': 'e.g. white ceramic vase',
        'instruction': (
            '''Return 3-6 words that identify the tangible objects or surfaces in the source image that should be recolored in the simplest sense, no need to be specific if there is no room for identification error. 
            Include their current color. Exclude filament holder.'''
        ),
    },
    {
        'name': 'reference_colors',
        'role': 'reference',
        'label': 'Reference colors',
        'placeholder': 'e.g. warm copper and teal',
        'instruction': (
            'Generate in 3-6 words a description of the colour and texture in the reference image. Do not describe the object just the colour tone and texture for it.'
        ),
    },
]

ROLE_ALIASES: Dict[str, str] = {
    'left': 'source',
    'source': 'source',
    'right': 'reference',
    'reference': 'reference',
}


def infer_role_from_name(name: str) -> str:
    """Infer a variable role from its name using common prefixes."""
    lowered = name.lower()
    for prefix, role in ROLE_ALIASES.items():
        if lowered.startswith(f'{prefix}_'):
            return role
    return 'source'


def _default_label(name: str) -> str:
    """Create a user-facing label from a variable name."""
    return re.sub(r'[_\-]+', ' ', name).strip().title()


def normalize_variable_definition(definition: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a variable definition payload."""
    name = str(definition.get('name', '')).strip()
    if not name:
        raise ValueError('Variable definitions must include a non-empty "name" field.')
    role = str(definition.get('role') or infer_role_from_name(name)).strip().lower()
    if role not in ('source', 'reference'):
        role = infer_role_from_name(name)
    label = str(definition.get('label') or _default_label(name)).strip()
    placeholder = str(definition.get('placeholder', '')).strip()
    instruction = str(definition.get('instruction', '')).strip() or f'Provide a concise description for "{label}".'

    return {
        'name': name,
        'role': role,
        'label': label,
        'placeholder': placeholder,
        'instruction': instruction,
    }


def normalize_definitions(definitions: Optional[Iterable[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalize a collection of variable definitions."""
    if not definitions:
        return [normalize_variable_definition(item) for item in DEFAULT_VARIABLE_DEFINITIONS]
    allowed = {item['name'] for item in DEFAULT_VARIABLE_DEFINITIONS}
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for entry in definitions:
        try:
            normalized_entry = normalize_variable_definition(entry)
        except ValueError:
            continue
        if normalized_entry['name'] not in allowed or normalized_entry['name'] in seen:
            continue
        seen.add(normalized_entry['name'])
        normalized.append(normalized_entry)
    if not normalized:
        return [normalize_variable_definition(item) for item in DEFAULT_VARIABLE_DEFINITIONS]
    return normalized


def build_definition_lookup(definitions: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return a mapping of variable name to definition."""
    return {str(entry['name']): dict(entry) for entry in definitions}


def ensure_unique_variable_name(base: str, existing: Iterable[str]) -> str:
    """Generate a unique variable name given a base and existing names."""
    base_clean = re.sub(r'[^a-z0-9_]', '_', base.lower()).strip('_') or 'variable'
    if base_clean not in existing:
        return base_clean
    index = 2
    while f'{base_clean}_{index}' in existing:
        index += 1
    return f'{base_clean}_{index}'


def extract_variables_from_blocks(blocks: Optional[Sequence[Mapping[str, Any]]]) -> List[str]:
    """Extract unique variable names used in a block sequence."""
    if not blocks:
        return []
    seen: set[str] = set()
    variables: List[str] = []
    for block in blocks:
        if block.get('type') != 'variable':
            continue
        name = str(block.get('name', '')).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        variables.append(name)
    return variables


def variables_by_role(
    blocks: Optional[Sequence[Mapping[str, Any]]],
    definitions: Sequence[Mapping[str, Any]],
) -> Dict[str, List[str]]:
    """Return variables used in the blocks grouped by role."""
    lookup = build_definition_lookup(definitions)
    grouped: Dict[str, List[str]] = {'source': [], 'reference': []}
    for name in extract_variables_from_blocks(blocks):
        role = lookup.get(name, {}).get('role') or infer_role_from_name(name)
        if role not in grouped:
            continue
        grouped[role].append(name)
    return grouped


def get_variable_definition(definitions: Sequence[Mapping[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single variable definition by name."""
    lookup = build_definition_lookup(definitions)
    return lookup.get(name)


def build_instruction_for_variables(
    variable_names: Sequence[str],
    definitions: Sequence[Mapping[str, Any]],
) -> str:
    """Create a human-readable instruction block for a set of variables."""
    if not variable_names:
        return ''
    lookup = build_definition_lookup(definitions)
    parts: List[str] = []
    for name in variable_names:
        definition = lookup.get(name)
        label = definition.get('label') if definition else _default_label(name)
        instruction = definition.get('instruction') if definition else f'Provide a concise description for "{label}".'
        parts.append(f'"{name}": {instruction}')
    return '; '.join(parts)
