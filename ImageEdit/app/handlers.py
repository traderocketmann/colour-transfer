"""Event handlers for the Color Transfer Assistant."""

import uuid

from nicegui import events, ui

from models import ImageDescription, UploadedImage, compress_image


def make_image_column(role: str, state):
    """Create image column with upload button and image cards."""
    role_label = 'Source' if role == 'source' else 'Reference'

    # Container for holding both upload button and image cards
    with ui.column().classes('w-full gap-2'):
        async def handle_upload(event: events.UploadEventArguments) -> None:
            import os

            try:
                data = await event.file.read()
            except Exception as error:
                ui.notify(f'Failed to read uploaded file: {error}', color='negative')
                return

            # Generate compressed thumbnail and save to disk
            thumbnail_path = compress_image(data, event.file.content_type)

            # Save full image to server
            image_id = uuid.uuid4().hex
            file_ext = event.file.name.split('.')[-1] if '.' in event.file.name else 'png'
            full_image_path = f'/app/uploads/{image_id}.{file_ext}'
            with open(full_image_path, 'wb') as f:
                f.write(data)

            default_description = ImageDescription(id=uuid.uuid4().hex, text='')
            new_image = UploadedImage(
                id=image_id,
                name=event.file.name,
                data=data,
                mime_type=event.file.content_type,
                thumbnail_path=thumbnail_path,
                full_image_path=full_image_path,
                descriptions=[default_description],
                selected_description_id=default_description.id,
            )

            # Add directly to state
            if role == 'source':
                state.source_images.append(new_image)
            else:
                state.reference_images.append(new_image)

            state.persist()
            ui.notify(f'Uploaded {event.file.name}', color='positive')
            state.render_image_column(role)

        # Hidden upload component
        upload = ui.upload(
            on_upload=handle_upload,
            multiple=True,
            auto_upload=True,
        ).classes('hidden')

        # Visible upload button that triggers the hidden upload
        ui.button(
            f'Upload {role_label.lower()} image(s)',
            on_click=lambda: upload.run_method('pickFiles'),
            icon='upload'
        ).props('outline')

        # Container for image cards
        container = ui.column().classes('w-full gap-4')
        state.register_image_container(role, container)


async def on_connect(_client, state) -> None:
    """Handle client connection event."""
    # Storage already loaded in index() before UI build
    # Only render if we have images (avoid unnecessary renders on empty state)
    if state.source_images or state.reference_images:
        state.render_all_images()
    if state.output_container is not None and state.output_images:
        from ui_components import render_output_gallery
        render_output_gallery(state.output_container, state.output_images, state)
    state.update_missing_button_visibility()

    # If job is running, show progress UI and update processing controls
    if state.is_processing():
        ui.notify('A processing job is currently running.', color='info')
        state.update_processing_controls()
        # Show progress bars and restore their values
        if state.overall_progress is not None:
            state.overall_progress.set_visibility(True)
            state.overall_progress.set_value(state.overall_progress_value)
        if state.workflow_progress is not None:
            state.workflow_progress.set_visibility(True)
            state.workflow_progress.set_value(state.workflow_progress_value)
        if state.workflow_progress_label is not None:
            state.workflow_progress_label.set_visibility(True)
            state.workflow_progress_label.set_text(state.workflow_progress_text)
