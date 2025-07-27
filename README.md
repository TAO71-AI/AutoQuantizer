# Installation
## 1. Install llama.cpp
```bash
git clone https://github.com/ggml-org/llama.cpp.git
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
|--repo-public|Make the created repository public.|-|False|
|--test|Test the script to make sure it works without executing commands.|-|False|
|--as-dir|Uploads the entire model directory in a single commit, instead of uploading files one by one.|-|False|

### LLM quantization methods
- **Q2_K**: Normal Q2\_K quantization. Most weights are in Q2\_K. Not recommended for most LLMs due to it's small precision.
- **Q2_K_L**: Uses Q8\_0 for the embedding and output weights, and mostly Q2\_K for everything else. Has more precision, but still it's not recommended because it's mostly Q2\_K.
- **Q2_K_XL**: Uses F16 for the embedding and output weights, and mostly Q2\_K for everything else. Has even more precision, but still not recommended because it's mostly Q2\_K.
- **Q3_K_S**: Normal Q3\_K\_S quantization. Most weights are in Q3\_K. Not recommended for most use cases due to it's small precision.
- **Q3_K_M**: Normal Q3\_K\_M quantization. There are more weights in other quantizations like Q5\_K and others, but mostly it is Q3\_K. Not recommended for most use cases due to it's small precision.
- **Q3_K_L**: Normal Q3\_K\_L quantization. There are even more weights in other quantizations, but mostly it is Q3\_K. If possible, prefer Q3\_K\_XL, but this might have decent results in some use cases. Only recommended if you have a very slow CPU, GPU, or RAM capacity.
- **Q3_K_XL**: Uses Q8\_0 for the embedding and output weights, and Q3\_K\_L for everything else. This might have decent results in some use cases.
- **Q3_K_XXL**: Uses F16 for the embedding and output weights, and Q3\_K\_L for everything else. Prefer this only if you want more precision for the embedding or output weights. For most models the size of this quant is similar to Q4\_K\_S or Q4\_K\_M. Prefer Q4\_K\_S or Q4\_K\_M if the size is similar.
- **Q4_K_S**: Normal Q4\_K\_S quantization. Most weights are in Q4\_K. Gives decent results for most use cases. Slightly lower quality than Q4\_K\_M and requires less CPU, GPU, and RAM.
- **Q4_K_M**: Normal Q4\_K\_M quantization. There are more weights in other quantizations like Q5\_K and others, but mostly it is Q4\_K. Gives decent results for most use cases. Good quality.
- **Q4_K_L**: Uses Q8\_0 for the embedding and output weights, and Q4\_K\_M for everything else. More precision than Q4\_K\_M.
- **Q4_K_XL**: Uses F16 for the embedding and output weights, and Q4\_K\_M for everything else. Prefer Q5\_K\_S or Q5\_K\_M if the size is similar.
- **Q5_K_S**: Normal Q5\_K\_S quantization. Most weights are in Q5\_K. High quality and very good results. Very similar to Q5\_K\_M but saving a bit more of memory.
- **Q5_K_M**: Normal Q5\_K\_M quantization. There are more weights in other quantizations like Q6\_K and others, but mostly it is Q5\_K. High quality and very good results.
- **Q5_K_L**: Uses Q8\_0 for the embedding and output weights, and Q5\_K\_M for everything else. High quality and very good results.
- **Q5_K_XL**: Uses F16 for the embedding and output weights, and Q5\_K\_M for everything else. Prefer Q6\_K if the size is similar.
- **Q6_K**: Normal Q6\_K quantization. Most weights are in Q6\_K. Very high quality. Results similar to Q8\_0.
- **Q6_K_L**: Uses Q8\_0 for the embedding and output weights, and mostly Q6\_K for everything else. Very high quality. Results more similar to Q8\_0 or F16.
- **Q6_K_XL**: Uses F16 for the embedding and output weights, and mostly Q6\_K for everything else. Prefer Q6\_K\_L.
- **Q8_0**: Normal Q8\_0 quantization. Most weights are in Q8\_0. Quality almost like F16, saving around half the memory required for F16.
- **Q8_K_XL**: Uses F16 for the embedding and output weights, and mostly Q8\_0 for everything else. Prefer Q8\_0.
- **F16**: Normal F16 quantization. Most weights are in F16. If the model has been trained in BF16 or F32, prefer BF16. Not recommended because Q8\_0 has almost the same quality. Only use this if the model has been trained in F16 and you really need full precision.
- **BF16**: Normal BF16 quantization. Most weights are in F16 or BF16. Not recommended because Q8\_0 has almost the same quality. Only use this if the model has been trained in BF16 or F32 and you really need full precision.
- **F32**: Normal F32 quantization. Most -if not all- weights are in F32. Not recommended because F16, BF16, and Q8\_0 has almost the same quality. Most LLMs are more than fine if you use F16, BF16, or Q8\_0. Only use this if the model has been trained in F32 and you really need full precision.

> [!NOTE]
> In most cases, Q8\_0 for embedding and output weights is enough, F16 doesn't make much difference.
