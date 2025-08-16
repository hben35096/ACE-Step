"""
ACE-Step: A Step Towards Music Generation Foundation Model

https://github.com/ace-step/ACE-Step

Apache 2.0 License
"""

import os
import click
import importlib.util
import yaml

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
    
with open("translations.yaml", 'r', encoding='utf-8') as f:
    translations= yaml.safe_load(f)


def localize_blocks(blocks, translations, language="zh"):
    translation_map = translations.get(language, {})

    for component in blocks.blocks.values():
        if hasattr(component, "label") and component.label in translation_map:
            component.label = translation_map[component.label]

        if hasattr(component, "info") and isinstance(component.info, str) and component.info in translation_map:
            component.info = translation_map[component.info]
        # 替换 Markdown / HTML 的 value
        if hasattr(component, "value") and isinstance(component.value, str):
            for key, value in translation_map.items():
                component.value = component.value.replace(key, value)
        # 如果是 Markdown，可以单独处理多行文本
        if hasattr(component, "markdown") and isinstance(component.markdown, str):
            for key, value in translation_map.items():
                component.markdown = component.markdown.replace(key, value)
 

@click.command()
@click.option(
    "--language",
    type=str,
    help="UI translation only supports Chinese for now, that is --language=zh",
)
@click.option(
    "--dark",
    type=click.BOOL,
    default=False,
    help="Enable dark UI mode",
)
@click.option(
    "--checkpoint_path",
    type=str,
    default="",
    help="Path to the checkpoint directory. Downloads automatically if empty.",
)
@click.option(
    "--server_name",
    type=str,
    default="127.0.0.1",
    help="The server name to use for the Gradio app.",
)
@click.option(
    "--port", type=int, default=7865, help="The port to use for the Gradio app."
)
@click.option("--device_id", type=int, default=0, help="The CUDA device ID to use.")
@click.option(
    "--share",
    type=click.BOOL,
    default=False,
    help="Whether to create a public, shareable link for the Gradio app.",
)
@click.option(
    "--bf16",
    type=click.BOOL,
    default=True,
    help="Whether to use bfloat16 precision. Turn off if using MPS.",
)
@click.option(
    "--torch_compile", type=click.BOOL, default=False, help="Whether to use torch.compile."
)
@click.option(
    "--cpu_offload", type=bool, default=False, help="Whether to use CPU offloading (only load current stage's model to GPU)"
)
@click.option(
    "--overlapped_decode", type=bool, default=False, help="Whether to use overlapped decoding (run dcae and vocoder using sliding windows)"
)
def main(checkpoint_path, server_name, port, device_id, share, bf16, torch_compile, cpu_offload, overlapped_decode, language, dark):
    """
    Main function to launch the ACE Step pipeline demo.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    spec = importlib.util.spec_from_file_location("gui", "acestep/ui/components.py")
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)

    # from acestep.ui.components import create_main_demo_ui
    from acestep.pipeline_ace_step import ACEStepPipeline
    from acestep.data_sampler import DataSampler

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode
    )
    data_sampler = DataSampler()
    
    ui_mode = js_func if dark else None

    demo = gui.create_main_demo_ui(
        text2music_process_func=model_demo.__call__,
        sample_data_func=data_sampler.sample,
        load_data_func=data_sampler.load_json,
        ui_mode=ui_mode,
    )
    if language:
        localize_blocks(demo, translations, language=language)
    demo.launch(server_name=server_name, server_port=port, share=share)


if __name__ == "__main__":
    main()
