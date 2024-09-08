# Development Environment Setup

<details><summary><h2>Setting up Virtual Environment</h2></summary>

* ### Option 1: VSC builtin
    * Create a VSC virtual environment with Ctrl + Shift + P -> Python: Create Environment
    * Select Python 3.11.9
* ### Option 2: Run the following commands
    * Create the virtual environment: ```python -m venv .venv```
    * Activate the virtual environment
        * On Windows: ```.\.venv\Scripts\activate```
        * On Mac: ```source .venv/bin/activate```
</details>

<details>
<summary><h2>Install requirements</h2></summary>

* #### Install requirements via requirements.txt
    * ```pip install -r requirements.txt```
    * NOTE: You may need to manually install some libraries that cause errors during installation.
* #### Setting up OpenAI API Token
    * Make an account on https://platform.openai.com
    * Obtain your API key, and register it on your system using the following command
    * ```export OPENAI_API_KEY="your_api_key_here"```
</details>

## File Structure

 * Regarding audio classification
     * data - contains AudioSet dataset files (including ontology)
     * models - checkpoint models are saved here
     * plots - training and testing plots are saved here
     * sounds_balanced_train_segments - contains all balanced dataset .flac audio files
     * sounds_eval_segments - contains all evaluation dataset .flac audio files

     * ``dataset.ipynb`` - dataset exploration helper
     * ``test.py`` - contains model testing code
     * ``train.py`` - contains model training code
 * Regarding audio/video descriptive AI
     * output - preprocessed frames from input video are saved here
     * videos - place the input.mp4 here, will also contain output.mp4 consisting of preprocessed frames

     * ``preprocess.ipynb`` - loads a model (either PANNs or a custom one) and places the timestamp & framewise audio classification on each frame
     * ``ai.ipynb`` - loads the preprocessed frames, and uses OpenAI API to create a audio-video content aware description
