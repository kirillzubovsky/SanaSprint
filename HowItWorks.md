# How Sana Sprint GUI Works

This document explains the setup process and the internal workings of the Sana Sprint GUI application.

## Setup (`setup.sh`)

The `setup.sh` script automates the initial setup of the application environment:

1.  **Virtual Environment:** It creates a dedicated Python 3.12 virtual environment named `.venv` in the project directory. This isolates the application's dependencies from other Python projects on the system.
    ```bash
    python3.12 -m venv .venv
    ```
2.  **Activation:** It activates the newly created virtual environment. Subsequent commands (like `pip install` and `python`) will operate within this isolated environment.
    ```bash
    source .venv/bin/activate
    ```
3.  **Dependency Installation:** It installs all the necessary Python packages specified in the `requirements.txt` file using `pip`. This includes libraries like `gradio`, `torch`, `diffusers`, etc.
    ```bash
    pip install -r requirements.txt
    ```
4.  **Application Launch:** Finally, it runs the main application script `app.py` using the Python interpreter within the activated virtual environment.
    ```bash
    python app.py
    ```

## Application Logic (`app.py`)

The `app.py` script contains the core logic for the Gradio interface and the text-to-image generation process:

1.  **Library Imports:** Imports necessary libraries:
    *   `gradio` for building the web UI.
    *   `torch` for tensor operations and device management.
    *   `diffusers` (from Hugging Face) for loading and using the pre-trained Sana Sprint models (`SanaSprintPipeline`).
    *   `numpy` and `random` for numerical operations and generating random seeds.
    *   `gc` (garbage collector) for memory management.

2.  **Device and Data Type Detection:**
    *   The script automatically detects the available hardware:
        *   NVIDIA GPU (CUDA)
        *   Apple Silicon GPU (MPS)
        *   CPU
    *   It selects the appropriate PyTorch data type (`dtype`) for optimal performance based on the detected device:
        *   `torch.bfloat16` if supported on CUDA.
        *   `torch.float16` for other CUDA devices or MPS.
        *   `torch.float32` for CPU.

3.  **Model Loading and Management:**
    *   Two Sana Sprint models are available: `0.6B` and `1.6B` parameters.
    *   The application defaults to the `0.6B` model on Apple Silicon (MPS) and `1.6B` otherwise.
    *   Models are loaded dynamically from Hugging Face Hub using `SanaSprintPipeline.from_pretrained(...)` when selected by the user.
    *   **Memory Optimization:** To conserve GPU/MPS memory, only one model is kept on the compute device (`cuda` or `mps`) at a time. When the user switches models:
        *   The currently loaded model (if any) is moved to the CPU (`.to("cpu")`).
        *   Its variable is set to `None` to allow Python's garbage collector to potentially free up memory.
        *   `gc.collect()` is called explicitly to encourage memory release.
        *   On MPS devices, `torch.mps.empty_cache()` is also called.
        *   The newly selected model is then loaded and moved to the compute device (`.to(device)`).
    *   A global variable `current_model_on_device` tracks which model (if any) is currently loaded on the GPU/MPS.

4.  **Inference Function (`infer`):**
    *   This function orchestrates the image generation process.
    *   It takes the user's prompt and selected parameters (model size, seed, width, height, guidance scale, inference steps) as input.
    *   It ensures the correct model pipeline (`pipe` for 0.6B, `pipe2` for 1.6B) is loaded onto the active device (handling model switching as described above).
    *   It sets up a random seed generator (`torch.Generator`) for reproducibility (or randomness if requested).
    *   It calls the selected `diffusers` pipeline with the provided parameters to generate the image.
    *   It returns the generated PIL image and the seed used.

5.  **Gradio User Interface:**
    *   A `gradio.Blocks` interface is created to provide a user-friendly web UI.
    *   **Components:**
        *   A text input field for the prompt.
        *   A "Run" button.
        *   An image output area to display the result.
        *   A radio button group to select the `model_size` ("0.6B" or "1.6B").
        *   An "Advanced Settings" accordion containing sliders and checkboxes for:
            *   `Seed`
            *   `Randomize seed` option
            *   `Width` and `Height` of the image
            *   `Guidance Scale`
            *   `Number of inference steps`
        *   Example prompts (`gr.Examples`) that users can click to try.
    *   **Interactivity:**
        *   The `infer` function is linked to the "Run" button's `click` event and the prompt's `submit` event (pressing Enter in the prompt box).
        *   The necessary inputs from the UI elements are passed to the `infer` function.
        *   The generated image and seed are displayed in the corresponding output components.

6.  **Launching the App:**
    *   `demo.launch()` starts the Gradio web server, making the interface accessible in a web browser. 