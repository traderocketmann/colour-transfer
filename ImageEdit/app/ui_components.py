"""UI rendering components for the Color Transfer Assistant."""

import base64
import binascii
import io
import os
import zipfile
from typing import Any, Dict, List, Optional

from nicegui import ui

from models import (
    UploadedImage,
    add_description_entry,
    get_variable_value,
    is_variable_filled,
    set_variable_value,
)
from prompt_settings import build_definition_lookup
from workflows import generate_description_for_image, retry_pair, process_with_custom_prompt


def get_image_bytes(entry: Dict[str, Any]) -> Optional[bytes]:
    """Extract image bytes from an output entry."""
    data = entry.get('bytes')
    if data is not None:
        return data
    full_image_path = entry.get('full_image_path')
    if full_image_path and os.path.exists(full_image_path):
        try:
            with open(full_image_path, 'rb') as file_handle:
                return file_handle.read()
        except Exception:
            return None
    data_url = entry.get('data_url')
    if not data_url:
        return None
    if ',' not in data_url:
        return None
    try:
        _, encoded = data_url.split(',', 1)
        return base64.b64decode(encoded)
    except (ValueError, binascii.Error):
        return None


def download_all_outputs(output_images: List[Dict[str, Any]]) -> None:
    """Download all output images as a zip file."""
    if not output_images:
        ui.notify('No output images to download.', color='warning')
        return

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for index, entry in enumerate(output_images, start=1):
            image_bytes = get_image_bytes(entry)
            if image_bytes is None:
                continue
            filename = entry.get('filename') or f'output-{index}.png'
            archive.writestr(filename, image_bytes)

    if buffer.tell() == 0:
        ui.notify('No downloadable image data found.', color='warning')
        return

    buffer.seek(0)
    ui.download(buffer.read(), 'outputs.zip', media_type='application/zip')




def render_output_gallery(container: ui.column, outputs: List[Dict[str, Any]], state) -> None:
    """Render the output image gallery organized by source image."""
    container.clear()
    with container:
        if not outputs:
            ui.label('No outputs yet.').classes('text-sm text-secondary')
            return

        # Group outputs by source_index
        from collections import defaultdict
        outputs_by_source = defaultdict(list)
        for entry in outputs:
            source_idx = entry.get('source_index', 0)
            outputs_by_source[source_idx].append(entry)

        # Render each row (one per source image)
        for source_idx in sorted(outputs_by_source.keys()):
            row_outputs = outputs_by_source[source_idx]
            with ui.row().classes('w-full gap-4 flex-wrap'):
                for entry in row_outputs:
                    entry_state = {'data': dict(entry)}

                    card = ui.card().classes('w-64 gap-2 items-start')
                    with card:
                        content = ui.column().classes('w-full gap-2')

                        def render_card(entry_data: Dict[str, Any]) -> None:
                            content.clear()
                            data_url = entry_data.get('data_url')
                            thumbnail_path = entry_data.get('thumbnail_path')
                            filename = entry_data.get('filename', f'output-{source_idx}.png')
                            mime_type = entry_data.get('mime_type') or 'image/png'
                            image_bytes = get_image_bytes(entry_data)
                            full_image_path = entry_data.get('full_image_path')

                            full_image_url = None
                            if full_image_path:
                                filename_only = full_image_path.split('/')[-1]
                                full_image_url = f'/downloads/{filename_only}'

                            if thumbnail_path:
                                filename_only = thumbnail_path.split('/')[-1]
                                thumbnail_url = f'/thumbnails/{filename_only}'
                            else:
                                thumbnail_url = None

                            display_url = thumbnail_url or full_image_url or data_url

                            with content:
                                preview_dialog = None
                                if display_url:
                                    ui.image(display_url).classes('w-full h-40 object-contain bg-gray-200 rounded')

                                # Show custom prompt badge and text if applicable
                                if entry_data.get('custom_prompt'):
                                    with ui.column().classes('w-full gap-1'):
                                        ui.label('Custom Prompt').classes('text-xs font-medium text-purple-700 bg-purple-100 px-2 py-1 rounded self-start')
                                        ui.label(f'"{entry_data.get("custom_prompt")}"').classes('text-xs text-secondary italic')

                                if full_image_url or data_url:
                                    with ui.dialog() as preview_dialog, ui.card().classes('max-w-4xl w-full gap-4 items-start'):
                                        ui.image(full_image_url or data_url).classes('w-full max-h-[80vh] object-contain bg-gray-200 rounded')
                                        ui.button('Close', on_click=preview_dialog.close).classes('self-end')
                                else:
                                    ui.label('⚠️ Thumbnail only - full image not stored').classes('text-xs text-orange-600')

                                async def handle_retry_pair(
                                    entry=entry_data,
                                    btn=None,
                                ):
                                    src_index = entry.get('source_index')
                                    ref_index = entry.get('reference_index')
                                    custom_prompt = entry.get('custom_prompt')

                                    # If this was generated with a custom prompt, use process_with_custom_prompt
                                    if custom_prompt:
                                        if btn is not None:
                                            btn.props('loading')
                                        await process_with_custom_prompt(state, entry, custom_prompt)
                                        if btn is not None:
                                            btn.props(remove='loading')
                                        return

                                    # Otherwise, use the standard retry_pair workflow
                                    if src_index is None or ref_index is None:
                                        ui.notify('Cannot retry this pair; missing pairing info.', color='warning')
                                        return
                                    if btn is not None:
                                        btn.props('loading')
                                    await retry_pair(state, int(src_index), int(ref_index))
                                    if btn is not None:
                                        btn.props(remove='loading')
                                    updated_entry = next(
                                        (
                                            item
                                            for item in state.output_images
                                            if item.get('source_index') == int(src_index)
                                            and item.get('reference_index') == int(ref_index)
                                        ),
                                        None,
                                    )
                                    if updated_entry:
                                        entry_state['data'] = dict(updated_entry)
                                        render_card(entry_state['data'])

                                with ui.row().classes('w-full justify-between items-center gap-2 flex-wrap'):
                                    if preview_dialog is not None:
                                        ui.button('View', on_click=preview_dialog.open).props('outline size=sm')
                                    else:
                                        ui.button('View', on_click=lambda: ui.notify('Full image not available', color='warning')).props('outline size=sm disabled')

                                    async def handle_download(
                                        file_path=full_image_path,
                                        name=filename,
                                        mime=mime_type,
                                    ):
                                        if not file_path or not os.path.exists(file_path):
                                            ui.notify('Image file unavailable.', color='warning')
                                            return
                                        try:
                                            with open(file_path, 'rb') as file_handle:
                                                ui.download(file_handle.read(), name, media_type=mime)
                                        except Exception as error:  # pragma: no cover - UI notification
                                            ui.notify(f'Failed to download image: {error}', color='negative')

                                    if full_image_path and os.path.exists(full_image_path):
                                        ui.button('Download', on_click=handle_download).props('outline size=sm')
                                    elif image_bytes is not None:
                                        ui.button(
                                            'Download',
                                            on_click=lambda data=image_bytes, name=filename, mime=mime_type: ui.download(
                                                data,
                                                name,
                                                media_type=mime,
                                            ),
                                        ).props('outline size=sm')
                                    else:
                                        ui.button('Download', on_click=lambda: ui.notify('Image data unavailable.', color='warning')).props('outline size=sm disabled')

                                    retry_btn = ui.button(
                                        icon='replay',
                                        on_click=lambda: None,
                                    ).props('flat round size=sm')

                                    # Set tooltip based on whether this was a custom prompt
                                    if entry_data.get('custom_prompt'):
                                        retry_btn.tooltip('Retry with same custom prompt')
                                    else:
                                        retry_btn.tooltip('Retry')

                                    retry_btn.on_click(lambda entry=entry_data, btn=retry_btn: handle_retry_pair(entry, btn))

                                # Custom prompt section
                                # Store draft prompts in state to survive re-renders
                                if not hasattr(state, '_draft_prompts'):
                                    state._draft_prompts = {}

                                # Use pair_index as unique key for this output
                                draft_key = str(entry_data.get('pair_index', '')) + '_' + str(entry_data.get('pair_number', ''))

                                # Initialize draft with existing custom prompt if available and not yet set
                                if draft_key not in state._draft_prompts and entry_data.get('custom_prompt'):
                                    state._draft_prompts[draft_key] = entry_data.get('custom_prompt')

                                current_draft = state._draft_prompts.get(draft_key, '')

                                with ui.expansion('Custom Prompt', icon='edit').classes('w-full q-mt-sm').props('dense'):
                                    with ui.column().classes('w-full gap-2'):
                                        ui.label('Post edit this image with a custom prompt').classes('text-xs text-secondary')

                                        custom_prompt_input = ui.textarea(
                                            placeholder='Enter custom prompt (e.g., "make the figurine slightly darker")',
                                            value=current_draft
                                        ).classes('w-full').props('outlined dense rows=3')

                                        def update_draft(e, key=draft_key):
                                            if not hasattr(state, '_draft_prompts'):
                                                state._draft_prompts = {}
                                            state._draft_prompts[key] = e.value

                                        custom_prompt_input.on('update:model-value', update_draft)

                                        async def handle_custom_prompt(
                                            entry=entry_data,
                                            key=draft_key,
                                            btn=None,
                                        ):
                                            prompt_text = state._draft_prompts.get(key, '').strip()
                                            if not prompt_text:
                                                ui.notify('Please enter a custom prompt.', color='warning')
                                                return
                                            if btn is not None:
                                                btn.props('loading')
                                            await process_with_custom_prompt(state, entry, prompt_text)
                                            if btn is not None:
                                                btn.props(remove='loading')

                                        custom_btn = ui.button(
                                            'Run with Custom Prompt',
                                            icon='play_arrow',
                                        ).props('color=primary size=sm')
                                        custom_btn.on_click(lambda entry=entry_data, key=draft_key, btn=custom_btn: handle_custom_prompt(entry, key, btn))

                        render_card(entry_state['data'])

        ui.button('Download all outputs', on_click=lambda: download_all_outputs(outputs)).props('outline').classes('q-mt-md')
def remove_image(role: str, image_id: str, state_obj) -> None:
    """Remove an image from the specified role."""
    import os
    from models import cleanup_thumbnail

    target_list = state_obj.source_images if role == 'source' else state_obj.reference_images
    index_to_remove = next((idx for idx, img in enumerate(target_list) if img.id == image_id), None)
    if index_to_remove is not None:
        image = target_list[index_to_remove]
        # Clean up thumbnail file
        cleanup_thumbnail(image.thumbnail_path)
        # Clean up full image file
        if image.full_image_path and os.path.exists(image.full_image_path):
            try:
                os.remove(image.full_image_path)
            except Exception:
                pass
        target_list.pop(index_to_remove)
        state_obj.persist()
    state_obj.render_image_column(role)


def render_image_column(role: str, container: ui.column, state) -> None:
    """Render the image column for the specified role.

    Args:
        role: 'source' or 'reference'
        container: The UI container to render into
        state: The ProcessingState instance (required if not importing from app)
    """
    images = state.source_images if role == 'source' else state.reference_images
    definitions_lookup = build_definition_lookup(state.ensure_variable_definitions())
    role_variables = [
        name for name in state.get_variables_for_role(role)
        if definitions_lookup.get(name, {}).get('role') == role
    ]
    if not role_variables:
        role_variables = [
            definition['name']
            for definition in state.ensure_variable_definitions()
            if definition.get('role') == role
        ]
    container.clear()

    if not images:
        with container:
            ui.label('No images uploaded yet.').classes('text-sm text-secondary')
            if not role_variables:
                ui.label(
                    'Use the “+ Variable” button above to define prompt variables for this column.'
                ).classes('text-xs text-secondary italic')
        return

    for idx, image in enumerate(images):
        if not image.descriptions:
            add_description_entry(image)

        with container:
            # Check if image has full data
            has_full_data = len(image.data) > 0
            missing_variables = [name for name in role_variables if not is_variable_filled(image, name)]

            with ui.card().classes('w-full gap-3 items-start'):
                with ui.row().classes('w-full justify-between items-start'):
                    ui.label(image.name).classes('text-sm font-medium truncate')
                    ui.button(
                        'Remove image',
                        on_click=lambda img_id=image.id, r=role: remove_image(r, img_id, state),
                    ).props('outline color=negative size=sm')

                # Warning for images without full data
                if not has_full_data:
                    ui.label('⚠️ Thumbnail only - re-upload for processing').classes('text-xs text-orange-600 font-medium')

                # Main content row: image preview and description controls
                with ui.row().classes('w-full gap-4 items-start flex-wrap'):
                    # Image preview (using thumbnail)
                    with ui.column().classes('gap-2'):
                        # Full resolution view dialog (only if we have full data)
                        if has_full_data:
                            with ui.dialog() as full_img_dialog, ui.card().classes('max-w-4xl w-full gap-4'):
                                ui.image(image.data_url).classes('w-full max-h-[80vh] object-contain bg-gray-200 rounded')
                                ui.button('Close', on_click=full_img_dialog.close).classes('self-end')

                            ui.image(image.thumbnail_url).classes('w-40 h-40 object-contain bg-gray-200 rounded flex-shrink-0 cursor-pointer').on('click', full_img_dialog.open)
                        else:
                            ui.image(image.thumbnail_url).classes('w-40 h-40 object-contain bg-gray-200 rounded flex-shrink-0 cursor-not-allowed opacity-75').on('click', lambda: ui.notify('Full image not available - please re-upload', color='warning'))

                    # Description controls
                    with ui.column().classes('flex-1 gap-3'):
                        if role_variables:
                            for variable_name in role_variables:
                                definition = definitions_lookup.get(variable_name, {})
                                label_text = variable_name.replace('_', ' ').title()
                                placeholder = definition.get('placeholder', '')
                                current_value = get_variable_value(image, variable_name)
                                missing_value = not current_value.strip()

                                field_column = ui.column().classes('w-full gap-1')
                                with field_column:
                                    ui.label(label_text).classes('text-xs font-semibold text-primary').tooltip(f'{{{variable_name}}}')

                                    def handle_value_change(event, img=image, name=variable_name):
                                        set_variable_value(img, name, event.value or '')
                                        state.persist()

                                    input_field = ui.input(
                                        value=current_value,
                                        placeholder=placeholder or None,
                                    ).classes('w-full')
                                    input_field.on_value_change(handle_value_change)
                                    input_field.props('dense outlined clearable')
                                    if definition.get('instruction'):
                                        input_field.tooltip(definition['instruction'])
                                    if missing_value:
                                        input_field.props('color=warning')
                        else:
                            ui.label(
                                'No prompt variables are linked to this column yet. Use the “+ Variable” button above to add one.'
                            ).classes('text-sm text-secondary italic')

                        has_counterpart = bool(state.reference_images) if role == 'source' else bool(state.source_images)

                        async def handle_generate(img=image, role_name=role, btn=None):
                            if btn is not None:
                                btn.props('loading')
                            state.log_message(f'Generating variables for {img.name}...')
                            await generate_description_for_image(state, role_name, img)
                            if btn is not None:
                                btn.props(remove='loading')
                            state.render_image_column(role_name)

                        buttons_row = ui.row().classes('w-full items-center gap-2 flex-wrap')
                        with buttons_row:
                            button_label = 'Generate variables'
                            generate_btn = ui.button(
                                button_label,
                                icon='auto_fix_high',
                            )
                            button_props = 'outline size=sm'
                            if missing_variables and role_variables and has_counterpart:
                                generate_btn.props(button_props)
                                generate_btn.on_click(
                                    lambda img=image, role_name=role, btn=generate_btn: handle_generate(img=img, role_name=role_name, btn=btn)
                                )
                                missing_text = ', '.join(f'{{{name}}}' for name in missing_variables)
                                generate_btn.tooltip(f'Fill missing variables: {missing_text}')
                            else:
                                generate_btn.props(f'{button_props} disabled')
                                if not has_counterpart:
                                    tooltip_text = (
                                        'Upload at least one reference image to generate source variables automatically.'
                                        if role == 'source'
                                        else 'Upload at least one source image to generate reference variables automatically.'
                                    )
                                    generate_btn.tooltip(tooltip_text)
                                elif not role_variables:
                                    generate_btn.tooltip('Add a variable to this column before generating values.')
                                else:
                                    generate_btn.tooltip('All variables are filled.')
