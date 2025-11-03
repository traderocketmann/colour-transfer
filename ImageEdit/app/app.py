"""Color Transfer Assistant - Main application entry point."""

import asyncio
import os
import uuid
from typing import Dict, Set

from dotenv import load_dotenv
from nicegui import app, context, ui

from clients import ComfyUIClient
from handlers import make_image_column, on_connect
from prompt_builder import render_prompt_builder
from state import ProcessingState
from ui_components import render_output_gallery
from workflows import (
    generate_missing_descriptions,
    one_click_color_transfer,
)

load_dotenv()


# Ensure storage directories exist
os.makedirs('/app/thumbnails', exist_ok=True)
os.makedirs('/app/uploads', exist_ok=True)
os.makedirs('/app/downloads', exist_ok=True)


# Configuration
STORAGE_SECRET = os.getenv('NICEGUI_STORAGE_SECRET')


# Session-scoped state registry
SESSION_STATES: Dict[str, ProcessingState] = {}
SESSION_CLIENTS: Dict[str, Set[int]] = {}


def _get_or_create_state(client=None) -> ProcessingState:
    """Retrieve or create the persistent state for the current browser session."""
    storage = app.storage.user
    session_id = storage.get('session_id')
    if not session_id:
        session_id = uuid.uuid4().hex
        storage['session_id'] = session_id

    state = SESSION_STATES.get(session_id)
    if state is None:
        state = ProcessingState()
        state.session_id = session_id
        state.comfy_client = ComfyUIClient()
        SESSION_STATES[session_id] = state
        SESSION_CLIENTS[session_id] = set()
    if client is not None:
        SESSION_CLIENTS.setdefault(session_id, set()).add(client.id)
    return state


def build_ui(state: ProcessingState) -> None:
    """Build the main UI for the provided processing state."""
    # Add SortableJS to the page
    ui.add_head_html('<script src="https://cdn.jsdelivr.net/npm/sortablejs@latest/Sortable.min.js"></script>')

    ui.markdown('# Color Transfer Assistant')
    ui.markdown(
        'Upload source and reference images, generate descriptions for each image, then transfer colors with ComfyUI.'
    )

    # Image upload sections
    with ui.row().classes('w-full items-start gap-6 flex-wrap'):
        with ui.column().classes('flex-1 min-w-[320px] gap-4'):
            with ui.row().classes('w-full items-center gap-2'):
                ui.markdown('### Source Images').classes('flex-grow')

                def handle_clear_source():
                    state.clear_source_images()
                    ui.notify('Source images cleared', color='positive')

                ui.button(
                    'Clear',
                    on_click=handle_clear_source,
                    icon='delete_sweep'
                ).props('outline color=negative size=sm').tooltip('Clear all source images')

            # Example prompts section for source
            with ui.expansion('Example Prompts', icon='lightbulb').classes('w-full').props('dense'):
                with ui.column().classes('w-full gap-3 q-pa-sm'):
                    ui.markdown('**Object Description Examples:**').classes('text-sm')

                    examples = [
                        {
                            'label': '1',
                            'text': 'both the green filament and figurine'
                        },
                        {
                            'label': '2',
                            'text': 'both the orange filament and statue'
                        },
                    ]

                    for example in examples:
                        with ui.row().classes('w-full gap-2 items-center'):
                            ui.label(f"{example['label']}:").classes('text-xs font-medium min-w-[140px]')
                            ui.label(f'"{example["text"]}"').classes('text-xs text-secondary italic')

                    ui.markdown('💡 **Tip:** Describe the object type, material, and surface characteristics in the source image.').classes('text-xs text-blue-700 q-mt-sm')

            make_image_column('source', state)

        with ui.column().classes('flex-1 min-w-[320px] gap-4'):
            with ui.row().classes('w-full items-center gap-2'):
                ui.markdown('### Reference Images').classes('flex-grow')

                def handle_clear_reference():
                    state.clear_reference_images()
                    ui.notify('Reference images cleared', color='positive')

                ui.button(
                    'Clear',
                    on_click=handle_clear_reference,
                    icon='delete_sweep'
                ).props('outline color=negative size=sm').tooltip('Clear all reference images')

            # Example prompts section
            with ui.expansion('Example Prompts', icon='lightbulb').classes('w-full').props('dense'):
                with ui.column().classes('w-full gap-3 q-pa-sm'):
                    ui.markdown('**Material Examples:**').classes('text-sm')

                    examples = [
                        {
                            'label': 'Cherry Red Filament',
                            'text': 'glossy bright faded-red'
                        },
                        {
                            'label': 'Transparent Blue Filament',
                            'text': 'glossy slightly faded midnight blue material'
                        },
                    ]

                    for example in examples:
                        with ui.row().classes('w-full gap-2 items-center'):
                            ui.label(f"{example['label']}:").classes('text-xs font-medium min-w-[140px]')
                            ui.label(f'"{example["text"]}"').classes('text-xs text-secondary italic')

                    ui.markdown('💡 **Tip:** Experiment with surface qualities (glossy/matte), brightness, saturation, and color names.').classes('text-xs text-blue-700 q-mt-sm')

            make_image_column('reference', state)

    # Prompt Builder Section
    with ui.column().classes('w-full gap-4 q-mt-lg'):
        render_prompt_builder(state)

    with ui.column().classes('w-full gap-2'):
        overall_progress = state.ensure_overall_progress()
        workflow_label = state.ensure_workflow_progress_label()
        workflow_progress = state.ensure_workflow_progress()
        overall_progress.set_value(0)
        workflow_progress.set_value(0)
        workflow_label.set_text('ComfyUI progress will appear here.')
        overall_progress.set_visibility(False)
        workflow_label.set_visibility(False)
        workflow_progress.set_visibility(False)

    # Control buttons
    with ui.row().classes('w-full items-center gap-4 flex-wrap q-mt-md'):
        async def handle_generate_missing() -> None:
            await generate_missing_descriptions(state)

        state.generate_missing_button = ui.button(
            'Generate missing variables',
            on_click=handle_generate_missing,
            icon='auto_awesome'
        ).props('outline').tooltip('Fill any missing prompt variables across all image pairs')
        state.generate_missing_button.visible = state.has_missing_descriptions()

        async def handle_one_click_color_transfer() -> None:
            if state.is_processing():
                ui.notify('Processing is already running.', color='warning')
                return
            try:
                await one_click_color_transfer(state)
            except asyncio.CancelledError:
                pass
            except Exception as error:
                state.log_message(f'Processing task failed: {error}')
                ui.notify('Processing encountered an unexpected error.', color='negative')

        one_click_button = ui.button(
            'One-click transfer',
            on_click=handle_one_click_color_transfer,
            icon='bolt'
        ).tooltip('Generate missing descriptions and run ComfyUI workflows')

        stop_button = ui.button(
            'Stop',
            on_click=state.request_stop,
            icon='stop'
        ).props('color=negative').tooltip('Stop the current job')
        stop_button.set_visibility(False)
        state.register_processing_controls(one_click_button, stop_button)

    # Log output
    state.set_log(ui.log().classes('w-full h-64 q-mt-md'))

    # Output gallery
    with ui.row().classes('w-full items-center gap-4'):
        ui.markdown('### Outputs').classes('flex-grow')

        def handle_clear_outputs():
            state.clear_outputs()
            state.persist()
            ui.notify('Output images cleared', color='positive')

        ui.button(
            'Clear output',
            on_click=handle_clear_outputs,
            icon='delete_sweep'
        ).props('outline color=negative').tooltip('Clear all output images and delete from server')

    output_container = ui.column().classes('w-full gap-4')
    state.set_output_container(output_container)
    render_output_gallery(output_container, state.output_images, state)


@ui.page('/')
def index() -> None:
    """Render the main application page for each client."""
    state = _get_or_create_state()
    # Load storage BEFORE building UI so initial render has data
    state.load_from_storage()
    state.reset_ui_elements()
    build_ui(state)


@app.on_connect
async def _on_connect(client) -> None:
    """Initialize per-client state after connection."""
    state = _get_or_create_state(client)
    await on_connect(client, state)


@app.on_disconnect
async def _on_disconnect(client) -> None:
    """Clean up per-client state on disconnect."""
    storage = client.storage
    session_id = storage.get('session_id')
    if not session_id:
        return
    state = SESSION_STATES.get(session_id)
    if state is None:
        return
    SESSION_CLIENTS.setdefault(session_id, set()).discard(client.id)
    if state.is_processing():
        state.log_message('Client disconnected; workflows continue running.')
    elif not SESSION_CLIENTS[session_id]:
        if state.comfy_client is not None:
            await state.comfy_client.close()
            state.comfy_client = None
        state.reset_ui_elements()


# Mount static directories for serving thumbnails, uploads, and downloads
app.add_static_files('/thumbnails', '/app/thumbnails')
app.add_static_files('/uploads', '/app/uploads')
app.add_static_files('/downloads', '/app/downloads')

ui.run(host='0.0.0.0', port=8000, reload=False, show=False, storage_secret=STORAGE_SECRET)
