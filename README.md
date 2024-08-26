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
    * Obtain your API key, and register it using the following command
    * ```export OPENAI_API_KEY="your_api_key_here"```
</details>