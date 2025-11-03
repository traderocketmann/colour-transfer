"""Prompt builder component for customizable prompt templates."""

import os
import uuid
import re
from typing import Dict, List

from nicegui import ui

from prompt_settings import build_definition_lookup, extract_variables_from_blocks

VARIABLE_PATTERN = re.compile(r'(\{[a-zA-Z0-9_\-]+\})')


def _parse_template_to_blocks(template: str) -> List[Dict[str, str]]:
    """Convert a prompt template string into builder blocks."""
    blocks: List[Dict[str, str]] = []
    parts = [part for part in VARIABLE_PATTERN.split(template) if part]
    for part in parts:
        if part.startswith('{') and part.endswith('}'):
            name = part[1:-1]
            blocks.append({
                "id": str(uuid.uuid4()),
                "type": "variable",
                "name": name,
            })
        else:
            blocks.append({
                "id": str(uuid.uuid4()),
                "type": "text",
                "content": part,
            })
    if not blocks:
        blocks.append({
            "id": str(uuid.uuid4()),
            "type": "text",
            "content": '',
        })
    return blocks


def get_default_prompt_blocks() -> List[Dict[str, str]]:
    """Generate default prompt blocks from environment configuration."""
    template = os.getenv('PROMPT_TEMPLATE')
    if not template:
        template = (
            'Edit the {source_objects} in the first image so that their '
            'colors, materials, and textures closely match the {reference_colors} from the second image while '
            'preserving context and composition.'
        )
    return _parse_template_to_blocks(template)


def blocks_to_template(blocks: List[Dict[str, str]]) -> str:
    """Convert blocks to a prompt template string."""
    parts = []
    for block in blocks or []:
        if block.get("type") == "text":
            parts.append(block.get("content", ""))
        elif block.get("type") == "variable":
            parts.append("{" + block.get("name", "") + "}")
    return " ".join(part for part in parts if part).strip()


def render_prompt_builder(state) -> None:
    """Render the prompt builder interface with a draggable variable bank."""
    if not hasattr(state, 'prompt_blocks') or state.prompt_blocks is None:
        state.prompt_blocks = get_default_prompt_blocks()
    state.ensure_variable_definitions()

    ui.markdown('### Prompt Template Builder')
    ui.label(
        'Arrange text and variable blocks to shape your prompt. Drag variables from the bank into the template.'
    ).classes('text-sm text-secondary q-mb-sm')

    builder_root = ui.column().classes('w-full gap-3')
    builder_area_id = f'prompt-builder-{uuid.uuid4().hex[:8]}'
    bank_area_id = f'prompt-bank-{uuid.uuid4().hex[:8]}'

    with builder_root:
        controls_row = ui.row().classes('w-full items-center justify-between gap-2 flex-wrap')

        def add_text_block(position: int | None = None) -> None:
            """Insert a new text block."""
            block = {
                'id': str(uuid.uuid4()),
                'type': 'text',
                'content': 'new text',
            }
            if position is None or position >= len(state.prompt_blocks):
                state.prompt_blocks.append(block)
            else:
                state.prompt_blocks.insert(position, block)
            state.persist()
            refresh_builder()

        with controls_row:
            ui.button(
                'Add text block',
                icon='text_fields',
                on_click=lambda: add_text_block(len(state.prompt_blocks)),
            ).props('outline size=sm')

            def reset_to_default() -> None:
                state.prompt_blocks = get_default_prompt_blocks()
                state.persist()
                refresh_builder()
                ui.notify('Prompt reset to default', color='positive')

            ui.button(
                'Reset to default',
                icon='refresh',
                on_click=reset_to_default,
            ).props('outline size=sm')

        ui.add_css(
            """
            .swap-highlight {
                border-color: #3b82f6 !important;
                background: rgba(59, 130, 246, 0.08) !important;
            }
            """
        )

        builder_area = ui.element('div').props(f'id="{builder_area_id}"').classes(
            'w-full flex flex-wrap items-start gap-2 border border-dashed border-gray-300 rounded bg-gray-50 p-2 min-h-[64px]'
        )
    bank_wrapper = ui.column().classes('w-full gap-1')
    with bank_wrapper:
        ui.label('Variable bank (drag to add)').classes('text-xs font-medium text-secondary uppercase tracking-wide')
        bank_area = ui.element('div').props(f'id="{bank_area_id}"').classes(
            'flex flex-wrap items-start gap-2'
        )
        ui.label(
            'Add new variables from the Source or Reference columns. Drag a variable into the template to insert its placeholder.'
        ).classes('text-xs text-secondary')

    ui.label('Current template:').classes('text-sm font-medium')
    template_preview = ui.label('').classes('text-sm text-secondary q-pa-sm bg-grey-2 rounded').style(
        'white-space: pre-wrap; word-break: break-word;'
    )

    def update_template_preview() -> None:
        template = blocks_to_template(state.prompt_blocks)
        state.prompt_template_text = template
        template_preview.set_text(template)

    def render_block(block: Dict[str, str], index: int, lookup: Dict[str, Dict]) -> None:
        """Render an individual block with inline sizing."""
        block_id = block.get('id') or str(uuid.uuid4())
        block['id'] = block_id
        block_type = block.get('type', 'text')

        container = ui.element('div').props(
            f'data-block-id="{block_id}" data-block-type="{block_type}"'
        ).classes(
            'prompt-block flex items-center gap-2 px-3 py-2 rounded border border-gray-300 bg-white shadow-sm select-none'
        )
        if block_type == 'variable':
            container.props(f'data-variable-name="{block.get("name", "")}"')

        with container:
            ui.icon('drag_indicator').classes('drag-handle cursor-grab text-grey-6').style('font-size: 20px;')

            if block_type == 'text':
                text_value = block.get('content', '')
                input_field = ui.input(
                    value=text_value,
                    placeholder='Add prompt text',
                ).props('dense outlined').classes('flex-1 min-w-[160px]')
                input_field.style('width: 100%;')

                def update_text(event, idx=index, widget=input_field):
                    value = event.value or ''
                    state.prompt_blocks[idx]['content'] = value
                    state.persist()
                    update_template_preview()

                input_field.on_value_change(update_text)
            else:
                var_name = block.get('name', '')
                display_label = var_name.replace('_', ' ').title()
                role_badge = 'Source' if lookup.get(var_name, {}).get('role', 'source') == 'source' else 'Reference'
                chip = ui.chip(
                    text=display_label,
                    color='primary',
                ).props('outline')
                chip.classes('text-sm font-medium px-2 py-1')
                badge = ui.label(role_badge)
                badge.classes('text-[10px] uppercase tracking-wide text-secondary font-semibold')

            def delete_block(idx=index):
                if 0 <= idx < len(state.prompt_blocks):
                    state.prompt_blocks.pop(idx)
                    state.persist()
                    refresh_builder()

            ui.button(
                icon='close',
                on_click=delete_block,
            ).props('flat round size=sm color=negative')

    def render_variable_bank(lookup: Dict[str, Dict]) -> None:
        """Render draggable variable chips."""
        bank_area.clear()
        active_variables = set(extract_variables_from_blocks(state.prompt_blocks))
        definitions = state.ensure_variable_definitions()
        name_to_definition = {definition['name']: definition for definition in definitions}
        ordered_definitions = [name_to_definition[name] for name in sorted(name_to_definition.keys())]

        with bank_area:
            for definition in ordered_definitions:
                name = definition['name']
                role = definition.get('role', 'source')
                item_classes = [
                    'prompt-block flex items-center gap-2 px-3 py-2 rounded border border-gray-300 bg-white shadow-sm select-none cursor-grab',
                ]
                if name in active_variables:
                    item_classes.append('border-primary-200 bg-primary-50')
                role_badge = 'Source' if role == 'source' else 'Reference'
                item = ui.element('div').props(
                    f'data-variable-name="{name}" data-block-type="variable"'
                ).classes(' '.join(item_classes))
                item.tooltip(definition.get('instruction', 'Drag into the template to insert this variable.'))
                with item:
                    ui.icon('drag_indicator').classes('drag-handle cursor-grab text-grey-6').style('font-size: 20px;')
                    display_label = name.replace('_', ' ').title()
                    chip = ui.chip(
                        text=display_label,
                        color='primary',
                    ).props('outline')
                    chip.classes('text-sm font-medium px-2 py-1')
                    badge = ui.label(role_badge)
                    badge.classes('text-[10px] uppercase tracking-wide text-secondary font-semibold')

    def refresh_builder() -> None:
        """Refresh blocks, bank, and preview."""
        definitions_local = state.ensure_variable_definitions()
        lookup = build_definition_lookup(definitions_local)
        builder_area.clear()
        with builder_area:
            if not state.prompt_blocks:
                placeholder = ui.element('div').classes(
                    'text-sm text-secondary italic px-2 py-1'
                )
                with placeholder:
                    ui.label('Add text blocks or drag variables here to start building your prompt.')
            else:
                for idx, block in enumerate(state.prompt_blocks):
                    render_block(block, idx, lookup)
        render_variable_bank(lookup)
        update_template_preview()
        setup_sortable()

    def handle_reorder(event) -> None:
        """Handle drag-and-drop reorder events from the frontend."""
        args = event.args or {}
        old_index = args.get('oldIndex')
        new_index = args.get('newIndex')
        if old_index is None or new_index is None or old_index == new_index:
            return
        try:
            block = state.prompt_blocks.pop(int(old_index))
        except (IndexError, ValueError):
            return
        state.prompt_blocks.insert(int(new_index), block)
        state.persist()
        refresh_builder()

    def handle_insert_variable(event) -> None:
        """Insert a variable block at the requested index."""
        args = event.args or {}
        name = args.get('name')
        index = args.get('index')
        if not name:
            return
        block = {
            'id': str(uuid.uuid4()),
            'type': 'variable',
            'name': name,
        }
        if isinstance(index, int) and 0 <= index <= len(state.prompt_blocks):
            state.prompt_blocks.insert(index, block)
        else:
            state.prompt_blocks.append(block)
        state.persist()
        refresh_builder()

    def setup_sortable() -> None:
        """Initialize SortableJS for the builder and bank containers."""
        ui.run_javascript(
            f"""
            (function() {{
                const builder = document.getElementById('{builder_area_id}');
                if (!builder || typeof Sortable === 'undefined') {{
                    return;
                }}
                if (!builder._sortableInstance) {{
                    builder._sortableInstance = Sortable.create(builder, {{
                        group: {{ name: 'prompt-builder', pull: true, put: true }},
                        animation: 150,
                        handle: '.drag-handle',
                        swap: true,
                        swapClass: 'swap-highlight',
                        onUpdate: function(evt) {{
                            emitEvent('reorder_blocks', {{ oldIndex: evt.oldIndex, newIndex: evt.newIndex }});
                        }},
                        onAdd: function(evt) {{
                            const name = evt.item && evt.item.getAttribute('data-variable-name');
                            const blockType = evt.item && evt.item.getAttribute('data-block-type');
                            const index = evt.newIndex;
                            if (name && blockType === 'variable') {{
                                evt.item.remove();
                                emitEvent('insert_variable_block', {{ name: name, index: index }});
                            }} else {{
                                evt.item.remove();
                            }}
                        }},
                    }});
                }}
                const bank = document.getElementById('{bank_area_id}');
                if (bank && !bank._sortableInstance) {{
                    bank._sortableInstance = Sortable.create(bank, {{
                        group: {{ name: 'prompt-builder', pull: 'clone', put: false }},
                        animation: 150,
                        sort: false,
                        handle: '.drag-handle',
                    }});
                }}
            }})();
            """
        )

    ui.on('reorder_blocks', handle_reorder)
    ui.on('insert_variable_block', handle_insert_variable)

    refresh_builder()
    state.prompt_builder_refresher = refresh_builder
