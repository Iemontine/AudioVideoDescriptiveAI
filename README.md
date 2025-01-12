# Embedded Temporal/Audio Context for Enhanced Video Content Description by LLM

* A hacky yet novel solution for providing additional temporal and audio context to LLM models capable of understanding visual data by embeddeding data directly into video frames.
* Attempted to identify sound effects present within video content via Resnet50 architecture trained on Log-mel spectrograms to classify diverse frequency patterns within underlying audio data.
* Resulted in noticably more detailed descriptions of videos, albeit susceptible to LLM (GPT-4o) hallucination. Audio classification models utilized suffer from 'cocktail party effect' issues.

### Formal research artifact can be found here: [Embedded TemporalAudio Context for Enhanced Video Content Description by LLM.pdf](https://github.com/user-attachments/files/18388456/Embedded.TemporalAudio.Context.for.Enhanced.Video.Content.Description.by.LLM.pdf)

## Results
### Proposed methodology for dataset ([Audioset](https://research.google.com/audioset/)) conversion to Log Mel Spectrogram
<img src="https://github.com/user-attachments/assets/640d08bb-1a75-452f-8dc8-6e1a6d208978" alt="proposed methodology" width="700" /></br>

### Training Results vs PANNs Utilization
<img src="https://github.com/user-attachments/assets/49624999-aaa3-4870-baa5-2f01e3dbcbd3" alt="Custom architecture training vs PANNs" width="700" /></br>

### Example of embedding
<img src="https://github.com/user-attachments/assets/8d287fea-e483-4041-a391-0715e5fa69cc" alt="Example embedding" width="700" /></br>


### Example output with embeddings
```
1 00:00:00,000 –> 00:00:01,333 A young girl sits on a couch, focused on her laptop.
2 00:00:01,333 –> 00:00:02,666 Rain pours heavily against the window.
3 00:00:02,666 –> 00:00:04,000 An explosion of sound surprises her, and she reacts.
4 00:00:04,000 –> 00:00:05,333 Thunder rumbles as she looks around.
5 00:00:05,333 –> 00:00:06,666 She turns away from the window, feeling uneasy.
6 00:00:06,666 –> 00:00:08,000 Outside the window, rain continues to fall.
..................
```
**These timings are accurate to data present within frames, and sound effects present on the frames as well.**


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
     * ``createVideo.ipynb`` - automatically edit the original video using the srt file and generate sound file using text-to-speech API.
