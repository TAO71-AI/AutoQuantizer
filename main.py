from huggingface_hub import snapshot_download
from jinja2 import Environment
import os
import shutil
import subprocess
import argparse
import re
import json
import yaml

DEFAULT_DOWNLOAD_DIR: str = "./downloads"
DEFAULT_OUTPUT_DIR: str = "./output"
DEFAULT_PYTHON_INTERPRETER: str = "python"
DEFAULT_LLAMA_CONVERT: str = "./llama.cpp/convert_hf_to_gguf.py"
DEFAULT_LLAMA_QUANTIZE: str = "./llama.cpp/build/bin/llama-quantize"
DEFAULT_MODEL_CARD: str = """

{%- if (mc_tags is string) -%}
    {{- "---\n" ~ mc_tags ~ "\n---\n\n" -}}
{%- endif -%}
{{- "# Quantizations\n\n" -}}
{%- if (has_mmproj) -%}
    {{- "## MMRPOJ\n\n" ~ mmproj_table ~ "\n\n## LLM\n\n" -}}
{%- endif -%}
{{- llm_table ~ "\n\n# AutoQuantizer\n\nThis model has been quantized using [TAO71-AI AutoQuantizer](https://github.com/TAO71-AI/AutoQuantizer)" -}}
{%- if (mc_content | length > 0) -%}
    {{- "\n\n# Original model card\n\nThe original model card will be provided below.\n\n---\n\n" ~ mc_content -}}
{%- endif -%}

""".strip()
DEFAULT_MODEL_CARD_QUANTIZATION_TABLE: str = """

{{- "|File name|File size|Quantization|\n|---|---|---|\n" -}}
{%- for quant in quants -%}
    {{- "|" ~ quant["file_name"] ~ "|" ~ quant["file_size"] ~ "|" ~ quant["quantization"] ~ "|" -}}
    {%- if (quants.index(quant) < quants | length - 1) -%}
        {{- "\n" -}}
    {%- endif -%}
{%- endfor -%}

""".strip()

def CheckDirectory(Dir: str) -> None:
    if (not os.path.exists(Dir) or not os.path.isdir(Dir)):
        os.mkdir(Dir)

def GetFileSize(FilePath: str) -> str:
    sizeBytes = os.path.getsize(FilePath)
    units = ["B", "KB", "MB", "GB"]
    idx = 0

    while (sizeBytes >= 1024 and idx < len(units) - 1):
        sizeBytes = sizeBytes / 1024
        idx += 1

    return f"{round(sizeBytes, 3)} {units[idx]}"

def ConvertSizeToBytes(Size: str) -> int:
    m = re.match(r"(\d+)\s*(GB|MB|KB|B)", Size, re.IGNORECASE)

    if (not m):
        return 0

    value = int(m.group(1))
    unit = m.group(2).upper()
    
    factors = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3
    }
    return value * factors.get(unit, 1)

if (__name__ == "__main__"):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", dest = "DOWNLOAD_DIR", required = False, default = DEFAULT_DOWNLOAD_DIR, type = str)
    parser.add_argument("--output-dir", dest = "OUTPUT_DIR", required = False, default = DEFAULT_OUTPUT_DIR, type = str)
    parser.add_argument("--python-interpreter", dest = "PY_INT", required = False, default = DEFAULT_PYTHON_INTERPRETER, type = str)
    parser.add_argument("--llama-convert", dest = "LCPP_CONVERT", required = False, default = DEFAULT_LLAMA_CONVERT, type = str)
    parser.add_argument("--llama-quantize", dest = "LCPP_QUANTIZE", required = False, default = DEFAULT_LLAMA_QUANTIZE, type = str)
    parser.add_argument("--hf-repo", dest = "HF_REPO", required = True, type = str)
    parser.add_argument("--llm-gguf-input", dest = "LLM_INPUT", required = False, default = None, type = str)
    parser.add_argument("--mmproj-gguf-input", dest = "MMPROJ_INPUT", required = False, default = None, type = str)
    parser.add_argument("--has-mmproj", dest = "HAS_MMPROJ", action = "store_true", required = False, default = False)
    parser.add_argument("--model-type", dest = "MODEL_TYPE", required = True, type = str)
    parser.add_argument("--mmproj-type", dest = "MMPROJ_TYPE", required = False, default = None, type = str)
    parser.add_argument("--quants", dest = "QUANTS", required = False, default = [], type = str, nargs = "+")
    parser.add_argument("--exit-on-quant-error", dest = "RAISE_QUANT_LLM", action = "store_true", required = False, default = True)
    parser.add_argument("--quants-mmproj", dest = "MMPROJ_QUANTS", required = False, default = [], type = str, nargs = "+")
    parser.add_argument("--exit-on-mmproj-quant-error", dest = "RAISE_QUANT_MMPROJ", action = "store_true", required = False, default = True)
    parser.add_argument("--extra-quantizations", dest = "EXTRA_QUANTS", required = False, default = "{}", type = str)
    parser.add_argument("--quant-threads", dest = "QUANT_THREADS", required = False, default = None, type = int)
    parser.add_argument("--quant-imatrix", dest = "QUANT_IMATRIX", required = False, default = None, type = str)
    parser.add_argument("--no-model-card", dest = "NO_MODEL_CARD", action = "store_true", required = False, default = False)
    parser.add_argument("--model-card-template", dest = "MODEL_CARD_TEMPLATE", required = False, default = DEFAULT_MODEL_CARD, type = str)
    parser.add_argument("--model-card-table-template", dest = "MODEL_CARD_TABLE_TEMPLATE", required = False, default = DEFAULT_MODEL_CARD_QUANTIZATION_TABLE, type = str)

    args = parser.parse_args()
    mmprojQuants = []
    llmQuants = []

    # Create directories
    downloadDir = args.DOWNLOAD_DIR
    outputDir = args.OUTPUT_DIR

    CheckDirectory(downloadDir)
    CheckDirectory(outputDir)

    # Download model
    repoUser = args.HF_REPO[:args.HF_REPO.find("/")]
    repoName = args.HF_REPO[args.HF_REPO.find("/") + 1:]
    downloadDir = f"{downloadDir}/{repoUser}_{repoName}"

    snapshot_download(repo_id = args.HF_REPO, repo_type = "model", local_dir = downloadDir)

    # Convert to GGUF
    modelOutputDir = f"{outputDir}/{repoUser}_{repoName}"
    CheckDirectory(modelOutputDir)

    modelOutputFile = f"{modelOutputDir}/{repoUser}_{repoName}.{args.MODEL_TYPE}.gguf"
    mmprojType = args.MODEL_TYPE if (args.MMPROJ_TYPE is None) else args.MMPROJ_TYPE
    mmprojOutputFile = f"{modelOutputDir}/mmproj.{repoUser}_{repoName}.{mmprojType}.gguf"

    if (args.LLM_INPUT is None and (not os.path.exists(modelOutputFile) or not os.path.isfile(modelOutputFile))):
        conversionProcess = subprocess.Popen([args.PY_INT, args.LCPP_CONVERT, downloadDir, "--outfile", modelOutputFile, "--outtype", args.MODEL_TYPE.lower()])
        conversionProcessOut = conversionProcess.wait()

        if (conversionProcessOut != 0):
            raise ChildProcessError("Could not convert model (LLM) due to an error.")
    elif (args.LLM_INPUT is not None and (not os.path.exists(modelOutputFile) or not os.path.isfile(modelOutputFile))):
        shutil.copy(args.LLM_INPUT, modelOutputFile)

    if (args.QUANT_IMATRIX is None):
        imatrixOutputFile = None
    else:
        imatrixOutputFile = f"{modelOutputDir}/imatrix.{repoUser}_{repoName}.gguf"

        if (not os.path.exists(imatrixOutputFile)):
            shutil.copy(args.QUANT_IMATRIX, imatrixOutputFile)

        llmQuants.append({"file_name": imatrixOutputFile.split("/")[-1], "file_size": GetFileSize(imatrixOutputFile), "quantization": "imatrix"})

    llmQuants.append({"file_name": modelOutputFile.split("/")[-1], "file_size": GetFileSize(modelOutputFile), "quantization": args.MODEL_TYPE})

    if (args.MMPROJ_INPUT is None and args.HAS_MMPROJ and (not os.path.exists(mmprojOutputFile) or not os.path.isfile(mmprojOutputFile))):
        conversionProcess = subprocess.Popen([args.PY_INT, args.LCPP_CONVERT, downloadDir, "--outfile", mmprojOutputFile, "--outtype", mmprojType.lower(), "--mmproj"])
        conversionProcessOut = conversionProcess.wait()

        if (conversionProcessOut != 0):
            raise ChildProcessError("Could not convert model (MMPROJ) due to an error.")
    elif (args.MMPROJ_INPUT is not None and args.HAS_MMPROJ and (not os.path.exists(mmprojOutputFile) or not os.path.isfile(mmprojOutputFile))):
        shutil.copy(args.MMPROJ_INPUT, mmprojOutputFile)

    if (args.HAS_MMPROJ):
        mmprojQuants.append({"file_name": mmprojOutputFile.split("/")[-1], "file_size": GetFileSize(mmprojOutputFile), "quantization": mmprojType})

    # Parse extra quantizations
    extraQuants = json.loads(args.EXTRA_QUANTS)

    # Quantize LLM
    for quantName in args.QUANTS:
        quantType = (extraQuants[quantName]["type"] if (quantName in extraQuants) else quantName).lower()
        quantExtraArgs = extraQuants[quantName]["args"] if (quantName in extraQuants) else []
        modelQuantOutput = f"{modelOutputDir}/{repoUser}_{repoName}.{quantName}.gguf"
        processArgs = [args.LCPP_QUANTIZE] + (["--imatrix", args.QUANT_IMATRIX] if (args.QUANT_IMATRIX is not None) else []) + quantExtraArgs + [modelOutputFile, modelQuantOutput, quantType] + ([args.QUANT_THREADS] if (args.QUANT_THREADS is not None) else [])

        if (os.path.exists(modelQuantOutput) and os.path.isfile(modelQuantOutput)):
            llmQuants.append({"file_name": modelQuantOutput.split("/")[-1], "file_size": GetFileSize(modelQuantOutput), "quantization": quantName.upper()})
            continue

        quantProcess = subprocess.Popen(processArgs)
        quantProcessOut = quantProcess.wait()

        if (quantProcessOut != 0):
            if (args.RAISE_QUANT_LLM):
                raise ChildProcessError("Could not quantize model (LLM) due to an error.")
            else:
                print("Could not quantize model (LLM) due to an error.", flush = True)
                continue

        llmQuants.append({"file_name": modelQuantOutput.split("/")[-1], "file_size": GetFileSize(modelQuantOutput), "quantization": quantName.upper()})

    # Quantize MMPROJ
    if (args.HAS_MMPROJ):
        for quantName in args.MMPROJ_QUANTS:
            quantType = (extraQuants[quantName]["type"] if (quantName in extraQuants) else quantName).lower()
            quantExtraArgs = extraQuants[quantName]["args"] if (quantName in extraQuants) else []
            mmprojQuantOutput = f"{modelOutputDir}/mmproj.{repoUser}_{repoName}.{quantName}.gguf"
            processArgs = [args.LCPP_QUANTIZE] + quantExtraArgs + [mmprojOutputFile, mmprojQuantOutput, quantType] + ([args.QUANT_THREADS] if (args.QUANT_THREADS is not None) else [])

            if (os.path.exists(mmprojQuantOutput) and os.path.isfile(mmprojQuantOutput)):
                mmprojQuants.append({"file_name": mmprojQuantOutput.split("/")[-1], "file_size": GetFileSize(mmprojQuantOutput), "quantization": quantName.upper()})
                continue

            quantProcess = subprocess.Popen(processArgs)
            quantProcessOut = quantProcess.wait()

            if (quantProcessOut != 0):
                if (args.RAISE_QUANT_MMPROJ):
                    raise ChildProcessError("Could not quantize model (MMPROJ) due to an error.")
                else:
                    print("Could not quantize model (MMPROJ) due to an error.", flush = True)
                    continue

            mmprojQuants.append({"file_name": mmprojQuantOutput.split("/")[-1], "file_size": GetFileSize(mmprojQuantOutput), "quantization": quantName.upper()})

    # Sort quantization lists
    llmQuants = sorted(llmQuants, key = lambda x: x["file_name"])
    mmprojQuants = sorted(mmprojQuants, key = lambda x: x["file_name"])

    # Create model card
    if (not args.NO_MODEL_CARD):
        MC_Env = Environment()

        if (args.HAS_MMPROJ):
            MC_MmprojQuants = []

            MC_MmprojQuantTable = MC_Env.from_string(args.MODEL_CARD_TABLE_TEMPLATE).render(quants = llmQuants)
        else:
            MC_MmprojQuantTable = None

        MC_LlmQuantTable = MC_Env.from_string(args.MODEL_CARD_TABLE_TEMPLATE).render(quants = mmprojQuants)
        MC_Tags = {}
        MC_Content = ""

        if (MC_Content.startswith("---\n") and MC_Content.count("\n---") > 0):
            try:
                MC_Tags = yaml.safe_load(MC_Content[4:MC_Content.index("\n---")].strip())
            except:
                pass

            MC_Content = MC_Content[MC_Content.index("\n---") + 4:].strip()

        MC_Tags["base_model"] = [args.HF_REPO]

        MC_Template = MC_Env.from_string(args.MODEL_CARD_TEMPLATE)
        MC_RenderedTemplate = MC_Template.render(mc_tags = yaml.safe_dump(MC_Tags).strip(), mc_content = MC_Content, has_mmproj = args.HAS_MMPROJ, mmproj_table = MC_MmprojQuantTable, llm_table = MC_LlmQuantTable)

        with open(f"{modelOutputDir}/README.md", "w" if (os.path.exists(f"{modelOutputDir}/README.md") and os.path.isfile(f"{modelOutputDir}/README.md")) else "x") as f:
            f.write(MC_RenderedTemplate)
