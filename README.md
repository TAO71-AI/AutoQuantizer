# Installation
## 1. Install llama.cpp
```bash
git clone https://github.com/ggml-org/llama.cpp-git
cd llama.cpp
pip install -r requirements.txt
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release
cd ..
```

## 2. Install this script's requirements
```bash
pip install -r requirements.txt
```

# Usage
## LLMs quantization
```bash
python quantize_llm.py [ARGUMENTS]
```

|Argument|Description|Type|Default value|
|---|---|---|---|
|--repo=REPOSITORY|Set the model repository. *Required*|str|None|
|--outtype=QUANT|Set the outtype of the GGUF file. Do not include this quant in the **--quants** argument. *Required*|str|None|
|--gguf=FILE|Set a current GGUF file to quantize.|str|None|
|--quants="QUANT-1 QUANT-2 ..."|Set the list of quants to quantize the model with. Separated by spaces, the quant names must be valid. If not set, the model will only be converted to GGUF.|list (str separated by spaces)|""|
|--output-dir=DIRECTORY|Override the default output directory.|str|"output"|
|--cache-dir=DIRECTORY|Override the default cache directory.|str|"cache"|
|--lcpp-dir=DIRECTORY|Override the default llama.cpp directory.|str|"llama.cpp"|
|--lcpp-pre-gguf=COMMAND|Override the default command to execute when converting to GGUF.|str|"python"|
|--lcpp-gguf=FILE|Override the default script file to execute when converting to GGUF.|str|"convert\_hf\_to\_gguf.py"|
|--lcpp-pre-quant=COMMAND|Override the default command to execute when quantizing.|str|""|
|--lcpp-quant=FILE|Override the default script file to execute when quantizing.|str|"build/bin/llama-quantize"|
|--model-card-template=TEMPLATE|Override the default model card template.|str|Check the script.|
|--repo-name-template=TEMPLATE|Override the default repository name template.|str|Check the script.|
|--repo-public|Make the created repository public.|-|-|
|--test|Test the script to make sure it works without executing commands.|-|-|
