"""Workflow execution and description generation functions."""

import asyncio
import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from state import ProcessingState

from dotenv import load_dotenv
from nicegui import ui

from openai import OpenAI

from clients import ComfyUIClient, VisionClient, extract_output_text
from models import (
    UploadedImage,
    get_variable_value,
    is_variable_filled,
    set_variable_value,
)
from prompt_settings import build_definition_lookup, extract_variables_from_blocks

# Load environment variables
load_dotenv()


class MissingConfiguration(RuntimeError):
    """Raised when required runtime configuration is not available."""


# Configuration loading
WORKFLOW_PATH = Path(os.getenv('WORKFLOW_PATH') or 'workflows/replace_color.json')
PROMPT_TEMPLATE = os.getenv('PROMPT_TEMPLATE') or (
    'Edit the {source_objects} in the first image so that their '
    'colors, materials, and textures closely match the {reference_colors} from the second image while '
    'preserving context and composition.'
)
PROMPT_ENHANCER_MODEL = os.getenv('PROMPT_ENHANCER_MODEL') or 'gpt-4o-mini'
PROMPT_ENHANCER_INSTRUCTION = (
    '请将下面的图像处理提示润色并翻译成自然、专业的中文，用于颜色迁移任务。'
    '保持花括号中的变量占位符完全不变，语句应清晰、简洁。'
)
def load_workflow_template() -> Dict[str, Any]:
    """Load the ComfyUI workflow template from disk."""
    if not WORKFLOW_PATH.exists():
        raise FileNotFoundError(f'Workflow template not found at {WORKFLOW_PATH}')

    with WORKFLOW_PATH.open('r', encoding='utf-8') as file:
        return json.load(file)


WORKFLOW_TEMPLATE = load_workflow_template()


def require_openai_key() -> str:
    """Ensure an OpenAI API key is available."""
    openai_api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if openai_api_key:
        return openai_api_key

    raise MissingConfiguration('Please supply an OpenAI API key before processing.')


def _definition_for_variable(name: str, lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return a normalized variable definition for prompting the model."""
    definition = dict(lookup.get(name) or {})
    if not definition:
        definition = {
            'name': name,
            'label': name.replace('_', ' ').title(),
            'instruction': f'Provide a concise description for "{name}".',
        }
    definition.setdefault('name', name)
    definition.setdefault('instruction', f'Provide a concise description for "{definition.get("label", name)}".')
    return definition


def _known_values(image: UploadedImage, variables: List[str]) -> Dict[str, str]:
    """Collect already-filled variable values for context in the prompt."""
    result: Dict[str, str] = {}
    for name in variables:
        value = get_variable_value(image, name)
        if value:
            result[name] = value
    return result


def _variable_role(name: str, lookup: Dict[str, Dict[str, Any]]) -> str:
    """Determine whether a variable belongs to the source or reference image."""
    definition = lookup.get(name)
    if definition:
        role = definition.get('role')
        if role in {'source', 'reference'}:
            return role
    lowered = name.lower()
    if lowered.startswith(('left_', 'source_')):
        return 'source'
    if lowered.startswith(('right_', 'reference_')):
        return 'reference'
    return 'source'


async def _generate_variables_for_pair(
    state: ProcessingState,
    *,
    source_index: int,
    source_image: UploadedImage,
    reference_index: int,
    reference_image: UploadedImage,
    source_variables: List[str],
    reference_variables: List[str],
    definition_lookup: Dict[str, Dict[str, Any]],
) -> tuple[bool, Dict[str, str], Dict[str, str], Optional[Dict[str, Any]]]:
    """Generate missing variables for a single source/reference pair."""
    missing_source = [name for name in source_variables if not is_variable_filled(source_image, name)]
    missing_reference = [name for name in reference_variables if not is_variable_filled(reference_image, name)]

    if not missing_source and not missing_reference:
        return False, {}, {}, None

    left_requests = [_definition_for_variable(name, definition_lookup) for name in missing_source]
    right_requests = [_definition_for_variable(name, definition_lookup) for name in missing_reference]

    known_left = _known_values(source_image, source_variables)
    for name in missing_source:
        known_left.pop(name, None)
    known_right = _known_values(reference_image, reference_variables)
    for name in missing_reference:
        known_right.pop(name, None)

    result, usage_data = await state.vision_client.generate_variables(
        source_image,
        reference_image,
        left_requests=left_requests,
        right_requests=right_requests,
        known_left=known_left,
        known_right=known_right,
    )

    new_source_values: Dict[str, str] = {}
    new_reference_values: Dict[str, str] = {}
    for name, value in (result or {}).items():
        if not value:
            continue
        trimmed = value.strip()
        if not trimmed:
            continue
        if name in missing_source:
            set_variable_value(source_image, name, trimmed)
            new_source_values[name] = trimmed
        elif name in missing_reference:
            set_variable_value(reference_image, name, trimmed)
            new_reference_values[name] = trimmed

    return True, new_source_values, new_reference_values, usage_data


async def _generate_single_image_variables(
    state: ProcessingState,
    *,
    target_image: UploadedImage,
    counterpart_image: UploadedImage,
    variable_names: List[str],
    definition_lookup: Dict[str, Dict[str, Any]],
    role: str,
) -> tuple[bool, Dict[str, str], Dict[str, str], Optional[Dict[str, Any]]]:
    """Generate variables for a single image against a counterpart image."""
    if not variable_names:
        return False, {}, {}, None

    requests = [_definition_for_variable(name, definition_lookup) for name in variable_names]

    known_target = _known_values(target_image, variable_names)
    for name in variable_names:
        known_target.pop(name, None)

    placeholder = {
        name: f'[{name}]'
        for name in extract_variables_from_blocks(state.prompt_blocks)
        if name not in variable_names
    }

    if role == 'source':
        left_requests = requests
        right_requests = []
        known_left = known_target
        known_right = placeholder
    else:
        left_requests = []
        right_requests = requests
        known_left = placeholder
        known_right = known_target

    result, usage_data = await state.vision_client.generate_variables(
        target_image if role == 'source' else counterpart_image,
        counterpart_image if role == 'source' else target_image,
        left_requests=left_requests,
        right_requests=right_requests,
        known_left=known_left,
        known_right=known_right,
    )

    new_source_values: Dict[str, str] = {}
    new_reference_values: Dict[str, str] = {}

    for name, value in (result or {}).items():
        if not value:
            continue
        trimmed = value.strip()
        if not trimmed:
            continue
        if role == 'source':
            set_variable_value(target_image, name, trimmed)
            new_source_values[name] = trimmed
        else:
            set_variable_value(target_image, name, trimmed)
            new_reference_values[name] = trimmed

    return True, new_source_values, new_reference_values, usage_data


def _prepare_prompt_data(
    *,
    state: ProcessingState,
    template: str,
    template_variables: List[str],
    definitions_lookup: Dict[str, Dict[str, Any]],
    source_image: UploadedImage,
    reference_image: UploadedImage,
) -> Dict[str, Any]:
    """Build the formatted prompt and supporting metadata for a pair."""
    prompt_values: Dict[str, str] = {}
    missing_variables: List[str] = []

    for variable_name in template_variables:
        role = _variable_role(variable_name, definitions_lookup)
        if role == 'reference':
            value = get_variable_value(reference_image, variable_name)
        else:
            value = get_variable_value(source_image, variable_name)
        if not value:
            value = f'[{variable_name}]'
            missing_variables.append(variable_name)
        prompt_values[variable_name] = value

    try:
        prompt_text = template.format(**prompt_values)
    except KeyError as error:
        missing_key = str(error)
        state.log_message(f'Missing template variable {missing_key}; using fallback formatting.')
        prompt_text = template
        for key, value in prompt_values.items():
            prompt_text = prompt_text.replace('{' + key + '}', value)

    source_summary_parts = [
        prompt_values.get(name, '')
        for name in template_variables
        if _variable_role(name, definitions_lookup) == 'source'
    ]
    reference_summary_parts = [
        prompt_values.get(name, '')
        for name in template_variables
        if _variable_role(name, definitions_lookup) == 'reference'
    ]
    source_summary = '; '.join(part for part in source_summary_parts if part) or 'objects in the source image'
    reference_summary = '; '.join(part for part in reference_summary_parts if part) or 'colors, materials, and textures from the reference image'

    return {
        'prompt_text': prompt_text,
        'prompt_values': prompt_values,
        'missing_variables': missing_variables,
        'source_summary': source_summary,
        'reference_summary': reference_summary,
    }


async def _enhance_prompt_text(state: ProcessingState, prompt_text: str) -> str:
    """Return the prompt text unchanged (enhancement disabled)."""
    return prompt_text


async def _enhance_prompts(state: ProcessingState, prompts: List[str]) -> List[str]:
    """Enhance a batch of prompts concurrently."""
    if not prompts:
        return []
    tasks = [asyncio.create_task(_enhance_prompt_text(state, prompt)) for prompt in prompts]
    return await asyncio.gather(*tasks)


def make_workflow(
    left_image_name: str,
    right_image_name: str,
    prompt_text: str,
) -> Dict[str, Any]:
    """Create a ComfyUI workflow from the template with the specified parameters."""
    workflow = copy.deepcopy(WORKFLOW_TEMPLATE)

    # Update the positive prompt node.
    if '111' in workflow:
        workflow['111']['inputs']['prompt'] = prompt_text

    # Update load image nodes with freshly uploaded assets.
    if '78' in workflow:
        workflow['78']['inputs']['image'] = left_image_name
    if '106' in workflow:
        workflow['106']['inputs']['image'] = right_image_name

    # Randomize Seed
    if '3' in workflow:
        workflow['3']['inputs']['seed'] = random.randint(0,99999999)

    return workflow


def make_single_image_workflow(
    image_name: str,
    prompt_text: str,
) -> Dict[str, Any]:
    """Create a ComfyUI workflow for single-image processing with custom prompt.

    Uses the same image for both left and right slots, allowing custom prompt-based editing.
    """
    workflow = copy.deepcopy(WORKFLOW_TEMPLATE)

    # Update the positive prompt node.
    if '111' in workflow:
        workflow['111']['inputs']['prompt'] = prompt_text

    # Use the same image for both slots (second image left empty/unused)
    if '78' in workflow:
        workflow['78']['inputs']['image'] = image_name
    if '106' in workflow:
        # Leave this empty or use a placeholder - the workflow will use only the first image
        workflow['106']['inputs']['image'] = image_name

    # Randomize Seed
    if '3' in workflow:
        workflow['3']['inputs']['seed'] = random.randint(0,99999999)

    return workflow


async def generate_missing_descriptions(state: ProcessingState) -> None:
    """Generate values for any missing prompt variables across all image pairs."""
    state.log_message('Starting variable generation for missing fields...')

    if not state.source_images or not state.reference_images:
        state.log_message('Need at least one source and one reference image.')
        ui.notify('Need at least one source image and one reference image.', color='warning')
        return

    variable_groups = state.get_active_variables_by_role()
    source_variables = variable_groups.get('source', [])
    reference_variables = variable_groups.get('reference', [])

    if not source_variables and not reference_variables:
        state.log_message('Prompt template does not reference any variables to fill.')
        ui.notify('Prompt template does not reference any variables to fill.', color='info')
        return

    try:
        api_key = require_openai_key()
    except MissingConfiguration as error:
        state.log_message(f'Configuration error: {error}')
        ui.notify(str(error), color='negative')
        return

    if state.vision_client is None:
        state.vision_client = VisionClient(api_key=api_key)

    definition_lookup = build_definition_lookup(state.ensure_variable_definitions())

    pairs: List[tuple[int, UploadedImage, int, UploadedImage]] = [
        (source_idx, source_image, reference_idx, reference_image)
        for source_idx, source_image in enumerate(state.source_images)
        for reference_idx, reference_image in enumerate(state.reference_images)
    ]

    pairs_to_process = []
    for source_idx, source_image, reference_idx, reference_image in pairs:
        missing_source = [name for name in source_variables if not is_variable_filled(source_image, name)]
        missing_reference = [name for name in reference_variables if not is_variable_filled(reference_image, name)]
        if missing_source or missing_reference:
            pairs_to_process.append((source_idx, source_image, reference_idx, reference_image))

    if not pairs_to_process:
        state.log_message('All variables are already filled.')
        ui.notify('All variables are already filled.', color='info')
        state.update_missing_button_visibility()
        return

    progress = state.ensure_overall_progress()
    try:
        progress.set_value(0)
        progress.set_visibility(True)
    except RuntimeError:
        return

    total_pairs = len(pairs_to_process)
    completed = 0

    try:
        state.log_message(f'Generating variables for {total_pairs} pair(s)...')
        for pair_index, (source_idx, source_image, reference_idx, reference_image) in enumerate(pairs_to_process, start=1):
            try:
                called, new_source, new_reference, usage = await _generate_variables_for_pair(
                    state,
                    source_index=source_idx,
                    source_image=source_image,
                    reference_index=reference_idx,
                    reference_image=reference_image,
                    source_variables=source_variables,
                    reference_variables=reference_variables,
                    definition_lookup=definition_lookup,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                state.log_message(
                    f'Failed to generate variables for Source{source_idx + 1}-Reference{reference_idx + 1}: {error}'
                )
                ui.notify(
                    f'Failed to generate variables for pair {pair_index}.',
                    color='negative',
                )
                continue

            if not called:
                continue

            completed += 1
            state.log_message(
                f'Updated variables for Source{source_idx + 1}-Reference{reference_idx + 1}:'
            )
            for name, value in new_source.items():
                state.log_message(f'  Source {{{name}}}: {value}')
            for name, value in new_reference.items():
                state.log_message(f'  Reference {{{name}}}: {value}')
            if usage:
                state.log_message(f'  Cost: ${usage["cost"]:.6f}')

            state.persist()

            try:
                progress.set_value(completed / total_pairs)
            except RuntimeError:
                pass
    finally:
        try:
            progress.set_value(0)
            progress.set_visibility(False)
        except RuntimeError:
            pass

    state.update_missing_button_visibility()
    if completed:
        state.render_image_column('source')
        state.render_image_column('reference')
    ui.notify('Variable generation complete.', color='positive')

async def generate_description_for_image(state: ProcessingState, role: str, image: UploadedImage) -> Optional[Dict[str, Dict[str, str]]]:
    """Generate missing variables for a single image against all counterparts."""
    variable_groups = state.get_active_variables_by_role()
    source_variables = variable_groups.get('source', [])
    reference_variables = variable_groups.get('reference', [])

    if role not in {'source', 'reference'}:
        ui.notify('Unknown image role provided.', color='negative')
        return None

    if not source_variables and not reference_variables:
        ui.notify('Prompt template does not reference any variables to fill.', color='info')
        return None

    try:
        api_key = require_openai_key()
    except MissingConfiguration as error:
        ui.notify(str(error), color='negative')
        return None

    if state.vision_client is None:
        state.vision_client = VisionClient(api_key=api_key)

    definition_lookup = build_definition_lookup(state.ensure_variable_definitions())

    if role == 'source':
        try:
            source_index = next(idx for idx, item in enumerate(state.source_images) if item.id == image.id)
        except StopIteration:
            ui.notify('Source image not found.', color='warning')
            return None
        source_variables_to_fill = [name for name in source_variables if not is_variable_filled(image, name)]
        if not source_variables_to_fill:
            ui.notify('All source variables are already filled for this image.', color='info')
            return None
        reference_base = state.reference_images[0] if state.reference_images else image
        called, new_source, _, usage = await _generate_single_image_variables(
            state,
            target_image=image,
            counterpart_image=reference_base,
            variable_names=source_variables_to_fill,
            definition_lookup=definition_lookup,
            role='source',
        )
        if not called:
            ui.notify('No variables generated for this image.', color='info')
            return None
        for name, value in new_source.items():
            state.log_message(f'Updated source variable {{{name}}} for {image.name}: {value}')
        if usage:
            state.log_message(f'  Cost: ${usage["cost"]:.6f}')
    else:
        try:
            reference_index = next(idx for idx, item in enumerate(state.reference_images) if item.id == image.id)
        except StopIteration:
            ui.notify('Reference image not found.', color='warning')
            return None
        reference_variables_to_fill = [name for name in reference_variables if not is_variable_filled(image, name)]
        if not reference_variables_to_fill:
            ui.notify('All reference variables are already filled for this image.', color='info')
            return None
        source_base = state.source_images[0] if state.source_images else image
        _, new_reference, _, usage = await _generate_single_image_variables(
            state,
            target_image=image,
            counterpart_image=source_base,
            variable_names=reference_variables_to_fill,
            definition_lookup=definition_lookup,
            role='reference',
        )
        for name, value in new_reference.items():
            state.log_message(f'Updated reference variable {{{name}}} for {image.name}: {value}')
        if usage:
            state.log_message(f'  Cost: ${usage["cost"]:.6f}')

    state.persist()
    state.render_image_column('source')
    state.render_image_column('reference')
    state.update_missing_button_visibility()
    ui.notify('Variables updated for the selected image.', color='positive')
    return None


async def one_click_color_transfer(state: ProcessingState) -> None:
    """Run description generation for missing entries and then process pairs."""
    if state.is_processing():
        ui.notify('Processing is already running.', color='warning')
        return

    current_task = asyncio.current_task()
    if current_task is None:
        ui.notify('Unable to start processing task.', color='negative')
        return

    state.begin_processing(current_task)

    definitions_lookup = build_definition_lookup(state.ensure_variable_definitions())

    try:
        missing_before = state.has_missing_descriptions()
        if missing_before:
            await generate_missing_descriptions(state)
            if state.has_missing_descriptions():
                state.log_message('Descriptions are still missing; aborting one-click edit.')
                ui.notify('Descriptions are still missing. Please add them before processing.', color='warning')
                return

        await process_pairs(state)
    except asyncio.CancelledError:
        state.log_message('Processing was cancelled.')
        ui.notify('Processing stopped.', color='warning')
    except Exception as error:
        state.log_message(f'Processing failed: {error}')
        ui.notify(f'Processing failed: {error}', color='negative')
    finally:
        state.finish_processing()


async def _process_single_pair(
    state: ProcessingState,
    *,
    combo_index: int,
    pair_count: int,
    pair_number: int,
    source_idx: int,
    source_image: UploadedImage,
    reference_idx: int,
    reference_image: UploadedImage,
    prepared: Dict[str, Any],
    prompt_text: str,
) -> bool:
    """Run a ComfyUI workflow for a single source/reference pair."""
    if state.comfy_client is None:
        state.comfy_client = ComfyUIClient()

    pair_label = f'Source{source_idx + 1}-Reference{reference_idx + 1}'

    prompt_values: Dict[str, str] = prepared.get('prompt_values', {})
    missing_variables: List[str] = prepared.get('missing_variables', [])
    source_summary: str = prepared.get('source_summary', 'objects in the source image')
    reference_summary: str = prepared.get('reference_summary', 'colors, materials, and textures from the reference image')

    if missing_variables:
        state.log_message(
            f'Pair {combo_index}/{pair_count} ({pair_label}): using placeholder values for {", ".join(missing_variables)}.'
        )

    state.log_message(f'Pair {combo_index}/{pair_count} ({pair_label}): {prompt_text}')

    if state.workflow_progress is not None:
        try:
            state.workflow_progress.set_value(0)
        except RuntimeError:
            pass
    if state.workflow_progress_label is not None:
        try:
            state.workflow_progress_label.set_text(
                f'Pair {combo_index}/{pair_count} ({pair_label}): uploading images to ComfyUI...'
            )
        except RuntimeError:
            pass
    try:
        upload_timeout = float(os.getenv('COMFY_UPLOAD_TIMEOUT', '120'))
    except ValueError:
        upload_timeout = 120.0

    try:
        source_name = await asyncio.wait_for(
            state.comfy_client.upload_image(source_image),
            timeout=upload_timeout,
        )
        reference_name = await asyncio.wait_for(
            state.comfy_client.upload_image(reference_image),
            timeout=upload_timeout,
        )
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        state.log_message(f'Uploading images timed out for pair {pair_label}.')
        ui.notify(
            f'Upload timed out for pair {combo_index} ({pair_label}).',
            color='negative',
        )
        return False
    except Exception as error:
        state.log_message(f'Failed to upload images to ComfyUI: {error}')
        ui.notify(
            f'Upload failed for pair {combo_index} ({pair_label}).',
            color='negative',
        )
        return False

    workflow_payload = make_workflow(
        left_image_name=source_name,
        right_image_name=reference_name,
        prompt_text=prompt_text,
    )

    state.log_message(f'Queued workflow for pair {combo_index} ({pair_label}).')

    try:
        def _progress_handler(fraction: float, info: Dict[str, Any]) -> None:
            # Update and persist progress values
            state.update_workflow_progress(fraction)
            node_name = info.get('node')
            percent = fraction * 100.0
            if node_name:
                text = f'Pair {combo_index}/{pair_count} ({pair_label}): node {node_name} {percent:.1f}%'
            else:
                text = f'Pair {combo_index}/{pair_count} ({pair_label}): {percent:.1f}% complete'
            state.update_workflow_progress_text(text)

        result = await state.comfy_client.run_workflow(
            workflow_payload,
            progress_callback=_progress_handler,
        )
        if isinstance(result, dict):
            streamed_images = result.get('websocket_images', [])
            if streamed_images:
                state.log_message(f'Received {len(streamed_images)} image(s) via websocket stream.')
                for img_idx, image_info in enumerate(streamed_images, start=1):
                    filename = image_info.get('filename') or f'pair-{pair_number}-{img_idx}.png'
                    image_bytes = image_info.get('bytes')
                    mime_type = image_info.get('mime_type') or 'image/png'
                    data_url = image_info.get('data_url')
                    state.add_output_image(
                        {
                            'pair_index': pair_label,
                            'filename': filename,
                            'data_url': data_url,
                            'bytes': image_bytes,
                            'mime_type': mime_type,
                            'pair_number': pair_number,
                            'source_description': source_summary,
                            'reference_description': reference_summary,
                            'source_index': source_idx,
                            'reference_index': reference_idx,
                        }
                    )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        state.log_message(f'Workflow failed: {error}')
        ui.notify(
            f'Workflow failed for pair {combo_index} ({pair_label}).',
            color='negative',
        )
        return False

    if state.workflow_progress is not None:
        try:
            state.workflow_progress.set_value(1)
        except RuntimeError:
            pass
    if state.workflow_progress_label is not None:
        try:
            state.workflow_progress_label.set_text(
                f'Pair {combo_index}/{pair_count} ({pair_label}): ComfyUI workflow complete.'
            )
        except RuntimeError:
            pass
    return True


async def process_pairs(state: ProcessingState) -> None:
    """Process pairs using existing descriptions to run ComfyUI workflows."""
    if not state.source_images or not state.reference_images:
        ui.notify('Upload at least one source image and one reference image.', color='warning')
        return

    # Check if all images have full data (not just thumbnails)
    missing_data_sources = [img.name for img in state.source_images if len(img.data) == 0]
    missing_data_refs = [img.name for img in state.reference_images if len(img.data) == 0]

    if missing_data_sources or missing_data_refs:
        error_msg = 'Some images are missing full data (thumbnail only). Please re-upload:\n'
        if missing_data_sources:
            error_msg += f'Sources: {", ".join(missing_data_sources)}\n'
        if missing_data_refs:
            error_msg += f'References: {", ".join(missing_data_refs)}'
        ui.notify(error_msg, color='negative', timeout=10000)
        state.log_message(error_msg)
        return

    pairs: List[tuple[int, UploadedImage, int, UploadedImage]] = [
        (source_idx, source_image, reference_idx, reference_image)
        for source_idx, source_image in enumerate(state.source_images)
        for reference_idx, reference_image in enumerate(state.reference_images)
    ]
    pair_count = len(pairs)
    if pair_count == 0:
        ui.notify('Nothing to process.', color='warning')
        return

    state.log_message(f'Processing {pair_count} pair(s) with ComfyUI workflows...')

    definitions_lookup = build_definition_lookup(state.ensure_variable_definitions())

    if state.prompt_blocks:
        from prompt_builder import blocks_to_template

        template = blocks_to_template(state.prompt_blocks)
        template_variables = extract_variables_from_blocks(state.prompt_blocks) or [
            'source_objects',
            'reference_colors',
        ]
    else:
        template = PROMPT_TEMPLATE
        template_variables = ['source_objects', 'reference_colors']

    def _get_overall_progress():
        progress_bar = state.overall_progress or state.ensure_overall_progress()
        return progress_bar

    def _prepare_workflow_widgets():
        progress_bar = state.workflow_progress or state.ensure_workflow_progress()
        label_widget = state.workflow_progress_label or state.ensure_workflow_progress_label()
        return progress_bar, label_widget

    overall_progress = _get_overall_progress()
    try:
        overall_progress.set_value(0)
        overall_progress.set_visibility(True)
    except RuntimeError:
        # Client disconnected
        return
    workflow_progress, workflow_progress_label = _prepare_workflow_widgets()
    try:
        workflow_progress_label.set_text('ComfyUI progress will appear here.')
        workflow_progress.set_value(0)
        workflow_progress.set_visibility(True)
        workflow_progress_label.set_visibility(True)
    except RuntimeError:
        # Client disconnected
        return
    state.clear_outputs()
    stopped_early = False

    try:
        if state.comfy_client is None:
            state.comfy_client = ComfyUIClient()

        references_count = max(1, len(state.reference_images))
        pair_infos: List[Dict[str, Any]] = []
        for combo_index, (source_idx, source_image, reference_idx, reference_image) in enumerate(pairs, start=1):
            prepared = _prepare_prompt_data(
                state=state,
                template=template,
                template_variables=template_variables,
                definitions_lookup=definitions_lookup,
                source_image=source_image,
                reference_image=reference_image,
            )
            pair_number = (source_idx * references_count) + reference_idx + 1
            pair_infos.append(
                {
                    'combo_index': combo_index,
                    'pair_count': pair_count,
                    'pair_number': pair_number,
                    'source_idx': source_idx,
                    'source_image': source_image,
                    'reference_idx': reference_idx,
                    'reference_image': reference_image,
                    'prepared': prepared,
                }
            )

        enhanced_prompts = await _enhance_prompts(
            state,
            [info['prepared']['prompt_text'] for info in pair_infos],
        )

        for info, prompt_text in zip(pair_infos, enhanced_prompts):
            if state.stop_requested:
                stopped_early = True
                state.log_message('Stop requested; aborting remaining pairs.')
                try:
                    workflow_progress_label.set_text('Processing stopped by user.')
                except RuntimeError:
                    pass
                break

            success = await _process_single_pair(
                state,
                combo_index=info['combo_index'],
                pair_count=pair_count,
                pair_number=info['pair_number'],
                source_idx=info['source_idx'],
                source_image=info['source_image'],
                reference_idx=info['reference_idx'],
                reference_image=info['reference_image'],
                prepared=info['prepared'],
                prompt_text=prompt_text,
            )
            state.update_overall_progress(info['combo_index'] / pair_count)
            if not success:
                continue

        if stopped_early or state.stop_requested:
            state.log_message('Processing stopped before completing all pairs.')
            try:
                workflow_progress_label.set_text('Processing stopped before completion.')
            except RuntimeError:
                pass
        else:
            try:
                overall_progress.set_value(1)
                ui.notify('Processing complete.')
                workflow_progress_label.set_text('All workflows complete.')
            except RuntimeError:
                pass
    except asyncio.CancelledError:
        stopped_early = True
        state.log_message('Processing cancelled.')
        try:
            workflow_progress_label.set_text('Processing cancelled.')
        except RuntimeError:
            pass
        raise
    finally:
        if state.comfy_client is not None:
            await state.comfy_client.close()
            state.comfy_client = None
        overall_progress = _get_overall_progress()
        workflow_progress, workflow_progress_label = _prepare_workflow_widgets()
        try:
            overall_progress.set_value(0)
            workflow_progress.set_value(0)
            workflow_progress_label.set_text('ComfyUI progress will appear here.')
            overall_progress.set_visibility(False)
            workflow_progress.set_visibility(False)
            workflow_progress_label.set_visibility(False)
        except RuntimeError:
            pass


async def process_with_custom_prompt(
    state: ProcessingState,
    output_entry: Dict[str, Any],
    custom_prompt: str,
) -> None:
    """Process an output image with a custom prompt (single-image mode)."""
    if state.is_processing():
        ui.notify('Processing is already running.', color='warning')
        return

    if not custom_prompt or not custom_prompt.strip():
        ui.notify('Please provide a custom prompt.', color='warning')
        return

    # Get the output image data from the entry
    image_bytes = output_entry.get('bytes')
    full_image_path = output_entry.get('full_image_path')

    # Try to get image bytes from the output entry
    if image_bytes is None and full_image_path and os.path.exists(full_image_path):
        try:
            with open(full_image_path, 'rb') as f:
                image_bytes = f.read()
        except Exception:
            pass

    if not image_bytes:
        ui.notify('Output image data not available.', color='warning')
        return

    # Get source index for metadata purposes
    source_index = output_entry.get('source_index', 0)

    current_task = asyncio.current_task()
    if current_task is None:
        ui.notify('Unable to start processing task.', color='negative')
        return

    state.begin_processing(current_task)

    def _get_overall_progress():
        progress_bar = state.overall_progress or state.ensure_overall_progress()
        return progress_bar

    def _prepare_workflow_widgets():
        progress_bar = state.workflow_progress or state.ensure_workflow_progress()
        label_widget = state.workflow_progress_label or state.ensure_workflow_progress_label()
        return progress_bar, label_widget

    overall_progress = _get_overall_progress()
    workflow_progress, workflow_progress_label = _prepare_workflow_widgets()
    try:
        overall_progress.set_value(0)
        overall_progress.set_visibility(True)
        workflow_progress.set_value(0)
        workflow_progress.set_visibility(True)
        workflow_progress_label.set_text('Processing with custom prompt...')
        workflow_progress_label.set_visibility(True)
    except RuntimeError:
        return

    try:
        if state.comfy_client is None:
            state.comfy_client = ComfyUIClient()

        output_filename = output_entry.get('filename', f'output-{source_index}.png')
        state.log_message(f'Processing output image {output_filename} with custom prompt: {custom_prompt}')

        # Upload the output image bytes
        try:
            upload_timeout = float(os.getenv('COMFY_UPLOAD_TIMEOUT', '120'))
        except ValueError:
            upload_timeout = 120.0

        try:
            mime_type = output_entry.get('mime_type', 'image/png')
            image_name = await asyncio.wait_for(
                state.comfy_client.upload_image_bytes(image_bytes, output_filename, mime_type),
                timeout=upload_timeout,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            state.log_message('Uploading image timed out.')
            ui.notify('Upload timed out.', color='negative')
            return
        except Exception as error:
            state.log_message(f'Failed to upload image to ComfyUI: {error}')
            ui.notify('Upload failed.', color='negative')
            return

        # Create single-image workflow
        workflow_payload = make_single_image_workflow(
            image_name=image_name,
            prompt_text=custom_prompt,
        )

        state.log_message('Queued custom prompt workflow.')

        def _progress_handler(fraction: float, info: Dict[str, Any]) -> None:
            state.update_workflow_progress(fraction)
            node_name = info.get('node')
            percent = fraction * 100.0
            if node_name:
                text = f'Custom prompt: node {node_name} {percent:.1f}%'
            else:
                text = f'Custom prompt: {percent:.1f}% complete'
            state.update_workflow_progress_text(text)

        result = await state.comfy_client.run_workflow(
            workflow_payload,
            progress_callback=_progress_handler,
        )

        if isinstance(result, dict):
            streamed_images = result.get('websocket_images', [])
            if streamed_images:
                state.log_message(f'Received {len(streamed_images)} image(s) via websocket stream.')
                for img_idx, image_info in enumerate(streamed_images, start=1):
                    filename = image_info.get('filename') or f'custom-{source_index}-{img_idx}.png'
                    image_bytes = image_info.get('bytes')
                    mime_type = image_info.get('mime_type') or 'image/png'
                    data_url = image_info.get('data_url')
                    state.add_output_image(
                        {
                            'pair_index': f'Custom-Source{source_index + 1}',
                            'filename': filename,
                            'data_url': data_url,
                            'bytes': image_bytes,
                            'mime_type': mime_type,
                            'pair_number': -1,
                            'source_description': custom_prompt,
                            'reference_description': 'Custom prompt (single image)',
                            'source_index': source_index,
                            'reference_index': -1,
                            'custom_prompt': custom_prompt,
                        }
                    )

        try:
            overall_progress.set_value(1)
            workflow_progress.set_value(1)
            workflow_progress_label.set_text('Custom prompt processing complete.')
            ui.notify('Custom prompt processing complete.', color='positive')
        except RuntimeError:
            pass

    except asyncio.CancelledError:
        state.log_message('Custom prompt processing cancelled.')
        try:
            workflow_progress_label.set_text('Processing cancelled.')
            ui.notify('Processing stopped.', color='warning')
        except RuntimeError:
            pass
    except Exception as error:
        state.log_message(f'Custom prompt processing failed: {error}')
        ui.notify(f'Processing failed: {error}', color='negative')
    finally:
        if state.comfy_client is not None:
            await state.comfy_client.close()
            state.comfy_client = None
        overall_progress = _get_overall_progress()
        workflow_progress, workflow_progress_label = _prepare_workflow_widgets()
        try:
            overall_progress.set_value(0)
            workflow_progress.set_value(0)
            workflow_progress_label.set_text('ComfyUI progress will appear here.')
            overall_progress.set_visibility(False)
            workflow_progress.set_visibility(False)
            workflow_progress_label.set_visibility(False)
        except RuntimeError:
            pass
        state.finish_processing()


async def retry_pair(state: ProcessingState, source_index: int, reference_index: int) -> None:
    """Retry a single source/reference pair workflow."""
    if state.is_processing():
        ui.notify('Processing is already running.', color='warning')
        return

    if (
        source_index is None
        or reference_index is None
        or source_index < 0
        or reference_index < 0
        or source_index >= len(state.source_images)
        or reference_index >= len(state.reference_images)
    ):
        ui.notify('Cannot retry this batch; original images are missing.', color='warning')
        return

    current_task = asyncio.current_task()
    if current_task is None:
        ui.notify('Unable to start processing task.', color='negative')
        return

    state.begin_processing(current_task)

    definitions_lookup = build_definition_lookup(state.ensure_variable_definitions())
    if state.prompt_blocks:
        from prompt_builder import blocks_to_template

        template = blocks_to_template(state.prompt_blocks)
        template_variables = extract_variables_from_blocks(state.prompt_blocks) or [
            'source_objects',
            'reference_colors',
        ]
    else:
        template = PROMPT_TEMPLATE
        template_variables = ['source_objects', 'reference_colors']

    def _get_overall_progress():
        progress_bar = state.overall_progress or state.ensure_overall_progress()
        return progress_bar

    def _prepare_workflow_widgets():
        progress_bar = state.workflow_progress or state.ensure_workflow_progress()
        label_widget = state.workflow_progress_label or state.ensure_workflow_progress_label()
        return progress_bar, label_widget

    overall_progress = _get_overall_progress()
    workflow_progress, workflow_progress_label = _prepare_workflow_widgets()
    try:
        overall_progress.set_value(0)
        overall_progress.set_visibility(True)
        workflow_progress.set_value(0)
        workflow_progress.set_visibility(True)
        workflow_progress_label.set_text('ComfyUI progress will appear here.')
        workflow_progress_label.set_visibility(True)
    except RuntimeError:
        # Client disconnected
        return

    references_count = max(1, len(state.reference_images))
    pair_number = (source_index * references_count) + reference_index + 1

    prepared = _prepare_prompt_data(
        state=state,
        template=template,
        template_variables=template_variables,
        definitions_lookup=definitions_lookup,
        source_image=state.source_images[source_index],
        reference_image=state.reference_images[reference_index],
    )

    try:
        enhanced_prompt = await _enhance_prompt_text(state, prepared['prompt_text'])
        success = await _process_single_pair(
            state,
            combo_index=1,
            pair_count=1,
            pair_number=pair_number,
            source_idx=source_index,
            source_image=state.source_images[source_index],
            reference_idx=reference_index,
            reference_image=state.reference_images[reference_index],
            prepared=prepared,
            prompt_text=enhanced_prompt,
        )
        if success:
            try:
                overall_progress.set_value(1)
                workflow_progress_label.set_text('Retry complete.')
                ui.notify('Batch retry complete.', color='positive')
            except RuntimeError:
                pass
        else:
            try:
                workflow_progress_label.set_text('Retry finished with errors.')
            except RuntimeError:
                pass
    except asyncio.CancelledError:
        state.log_message('Retry cancelled.')
        try:
            workflow_progress_label.set_text('Retry cancelled.')
            ui.notify('Retry stopped.', color='warning')
        except RuntimeError:
            pass
    except Exception as error:
        state.log_message(f'Retry failed: {error}')
        ui.notify(f'Retry failed: {error}', color='negative')
    finally:
        if state.comfy_client is not None:
            await state.comfy_client.close()
            state.comfy_client = None
        overall_progress = _get_overall_progress()
        workflow_progress, workflow_progress_label = _prepare_workflow_widgets()
        try:
            overall_progress.set_value(0)
            workflow_progress.set_value(0)
            workflow_progress_label.set_text('ComfyUI progress will appear here.')
            overall_progress.set_visibility(False)
            workflow_progress.set_visibility(False)
            workflow_progress_label.set_visibility(False)
        except RuntimeError:
            pass
        state.finish_processing()
