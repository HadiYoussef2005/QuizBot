# QuizBot

This is a bot that quizzes you on pdf notes, done through langchain utilizing the OpenAI LLM. Here is how you can run it.

### Installations
To run this, both python and pip must be installed, information on which can be found here:

python: https://www.python.org/downloads/

pip: https://pip.pypa.io/en/stable/installation/

Then, you will need to install the following python packages through your terminal

``` cmd
pip install python-dotenv
pip install streamlit
pip install pymupdf
pip install tiktoken
pip install faiss-cpu
pip install langchain
pip install OpenAI
```

### API Key
To run this code, you will need a file in the same directory called ".env" which contains the line
``` python
OPENAI_API_KEY=your_api_key_here
```
Where you will replace "your_api_key_here" with your api key

### Running
Finally, to actually get it running, in your terminal change directories to the directory that contains this project, and run "python -m streamlit run main.py"
