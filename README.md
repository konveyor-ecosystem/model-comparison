# Model Comparison

This repo is intended provide a simplified and more rapid way of producing results from various models for comparison than the notebooks at https://github.com/konveyor-ecosystem/kai/blob/main/notebooks

To see how the given prompt was assembled and what data went into it you can view the associated notebook.

## Usage
- Set up the python environment
```
python3 -m venv env
source env/bin/activate
MULTIDICT_NO_EXTENSIONS=1 pip3 install -r ./requirements.txt
```

- To run OpenAI models:
```
export OPENAI_API_KEY=...
```

- To run IBM models:
```
export GENAI_KEY=...
```

To run a model:
```
pushd jms_to_smallrye_reactive_few_shot
python ../compare.py
popd
```

The script attempts to add the necessary tags (`<s>[INST]<<SYS>><</SYS>>[/INST]`, etc.) where appropriate to a given model.

You can then examine the contents of files under `jms_to_smallrye_reactive_few_shot/output` to compare results from different models.
