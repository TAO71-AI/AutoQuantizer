import log as logging
from huggingface_hub import snapshot_download, create_repo, HfApi
from huggingface_hub.utils import RepositoryNotFoundError
import sys
import os
import shutil

def ExecuteCommand(Cmd: str, Critical: bool = False) -> None:
    logging.Log(logging.LOG_LEVEL_INFO, f"Executing command '{Cmd}'.")
    outCode = os.system(Cmd)

    if (outCode == 0):
        logging.Log(logging.LOG_LEVEL_INFO, "^-- Command completed without error.")
    elif (Critical):
        logging.Log(logging.LOG_LEVEL_CRIT, "^-- Error executing critical command.")
        exit(1)
    else:
        logging.Log(logging.LOG_LEVEL_ERRO, "^-- Error executing command.")

def GetQuantInfo(Name: str, Critical: bool = False) -> dict[str, str] | None:
    if (Name.upper() not in list(QUANTS_AVAILABLE.keys())):
        if (Critical):
            logging.Log(logging.LOG_LEVEL_CRIT, "Invalid quant name.")
            raise ValueError()
        else:
            return None

    return QUANTS_AVAILABLE[Name.upper()]

def FormatSize(SizeInBytes: int) -> tuple[int | float, str]:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if (SizeInBytes < 1024):
            return (round(SizeInBytes, 2), unit)
        
        SizeInBytes /= 1024

def GetRepoName() -> str:
    return LLM_CreateRepoNameTemplate.replace("[MODEL_NAME]", LLM_Repo.replace("/", "_"))

def RepoExists(Repo: str) -> bool:
    try:
        HF_API.repo_info(repo_id = Repo)
        return True
    except RepositoryNotFoundError:
        return False
    except Exception as ex:
        logging.Log(logging.LOG_LEVEL_ERRO, f"Error checking if a repository exists. Details: {ex}")

QUANTS_AVAILABLE: dict[str, dict[str, str]] = {
    "Q2_K": {
        "lcpp_name": None,
        "extra_params": "",
        "description": "Not recommended for most people. Very low quality."
    },
    "Q2_K_L": {
        "lcpp_name": "Q2_K",
        "extra_params": "--output-tensor-type Q8_0 --token-embedding-type Q8_0",
        "description": "Not recommended for most people. Uses Q8_0 for output and embedding, and Q2_K for everything else. Very low quality."
    },
    "Q2_K_XL": {
        "lcpp_name": "Q2_K",
        "extra_params": "--output-tensor-type F16 --token-embedding-type F16",
        "description": "Not recommended for most people. Uses F16 for output and embedding, and Q2_K for everything else. Very low quality."
    },
    "Q3_K_S": {
        "lcpp_name": None,
        "extra_params": "",
        "description": "Not recommended for most people. Prefer any bigger Q3_K quantization. Low quality."
    },
    "Q3_K_M": {
        "lcpp_name": None,
        "extra_params": "",
        "description": "Not recommended for most people. Low quality."
    },
    "Q3_K_L": {
        "lcpp_name": None,
        "extra_params": "",
        "description": "Not recommended for most people. Low quality."
    },
    "Q3_K_XL": {
        "lcpp_name": "Q3_K_L",
        "extra_params": "--output-tensor-type Q8_0 --token-embedding-type Q8_0",
        "description": "Not recommended for most people. Uses Q8_0 for output and embedding, and Q3_K_L for everything else. Low quality."
    },
    "Q3_K_XXL": {
        "lcpp_name": "Q3_K_L",
        "extra_params": "--output-tensor-type F16 --token-embedding-type F16",
        "description": "Not recommended for most people. Uses F16 for output and embedding, and Q3_K_L for everything else. Low quality."
    },
    "Q4_K_S": {
        "lcpp_name": None,
        "extra_params": "",
        "description": "Recommended. Slightly low quality."
    },
    "Q4_K_M": {
        "lcpp_name": None,
        "extra_params": "",
        "description": "Recommended. Decent quality for most use cases."
    },
    "Q4_K_L": {
        "lcpp_name": "Q4_K_M",
        "extra_params": "--output-tensor-type Q8_0 --token-embedding-type Q8_0",
        "description": "Recommended. Uses Q8_0 for output and embedding, and Q4_K_M for everything else. Decent quality."
    },
    "Q4_K_XL": {
        "lcpp_name": "Q4_K_M",
        "extra_params": "--output-tensor-type F16 --token-embedding-type F16",
        "description": "Recommended. Uses F16 for output and embedding, and Q4_K_M for everything else. Decent quality."
    },
    "Q5_K_S": {
        "lcpp_name": None,
        "extra_params": "",
        "description": "Recommended. High quality."
    },
    "Q5_K_M": {
        "lcpp_name": None,
        "extra_params": "",
        "description": "Recommended. High quality."
    },
    "Q5_K_L": {
        "lcpp_name": "Q5_K_M",
        "extra_params": "--output-tensor-type Q8_0 --token-embedding-type Q8_0",
        "description": "Recommended. Uses Q8_0 for output and embedding, and Q5_K_M for everything else. High quality."
    },
    "Q5_K_XL": {
        "lcpp_name": "Q5_K_M",
        "extra_params": "--output-tensor-type F16 --token-embedding-type F16",
        "description": "Recommended. Uses F16 for output and embedding, and Q5_K_M for everything else. High quality."
    },
    "Q6_K": {
        "lcpp_name": None,
        "extra_params": "",
        "description": "Recommended. Very high quality."
    },
    "Q6_K_L": {
        "lcpp_name": "Q6_K",
        "extra_params": "--output-tensor-type Q8_0 --token-embedding-type Q8_0",
        "description": "Recommended. Uses Q8_0 for output and embedding, and Q6_K for everything else. Very high quality."
    },
    "Q6_K_XL": {
        "lcpp_name": "Q6_K",
        "extra_params": "--output-tensor-type F16 --token-embedding-type F16",
        "description": "Recommended. Uses F16 for output and embedding, and Q6_K for everything else. Very high quality."
    },
    "Q8_0": {
        "lcpp_name": None,
        "extra_params": "",
        "description": "Recommended. Quality almost like F16."
    },
    "Q8_K_XL": {
        "lcpp_name": "Q8_0",
        "extra_params": "--output-tensor-type F16 --token-embedding-type F16",
        "description": "Recommended. Uses F16 for output and embedding, and Q8_0 for everything else. Quality almost like F16."
    },
    "F16": {
        "lcpp_name": None,
        "extra_params": "",
        "description": "Not recommended. Overkill. Prefer Q8_0."
    },
    "BF16": {
        "lcpp_name": None,
        "extra_params": "",
        "description": "Not recommended. Overkill. Prefer Q8_0."
    },
    "F32": {
        "lcpp_name": None,
        "extra_params": "",
        "description": "Not recommended. Overkill. Prefer Q8_0."
    }
}
Quants: list[str] = []
LLM_Repo: str | None = None
LLM_CreateRepoPrivate: bool = True
LLM_CreateRepoNameTemplate: str = "[MODEL_NAME]-GGUF"
LLM_CacheDir: str = "cache"
LLM_ModelCardTemplate: str = "[METADATA]\n[TABLE]\n\n---\n\nQuantized using [TAO71-AI AutoQuantizer](https://github.com/TAO71-AI/AutoQuantizer).\nYou can check out the original model card [here](https://huggingface.co/[IN_REPO])."
GGUF_Output: str = "output"
GGUF_OutType: str | None = None
GGUF_File: str | None = None
LLAMA_Dir: str = "llama.cpp"
LLAMA_PreConvertToGGUF: str = "python"
LLAMA_ConvertToGGUF: str = "convert_hf_to_gguf.py"
LLAMA_PreQuantize: str = ""
LLAMA_Quantize: str = "build/bin/llama-quantize"
TestingMode: bool = False
HF_API: HfApi = HfApi()
HF_Username: str = HF_API.whoami()["name"]

if (len(sys.argv) > 1):
    for arg in sys.argv[1:]:
        if (arg.startswith("--repo=")):
            LLM_Repo = arg[7:]
        elif (arg.startswith("--gguf=")):
            GGUF_File = arg[7:]
        elif (arg.startswith("--quants=")):
            Quants = arg[9:].split(" ")
        elif (arg.startswith("--output-dir=")):
            GGUF_Output = arg[13:]
        elif (arg.startswith("--cache-dir=")):
            LLM_CacheDir = arg[12:]
        elif (arg.startswith("--outtype=")):
            GGUF_OutType = arg[10:]
        elif (arg.startswith("--lcpp-dir=")):
            LLAMA_Dir = arg[11:]
        elif (arg.startswith("--lcpp-pre-gguf=")):
            LLAMA_PreConvertToGGUF = arg[16:]
        elif (arg.startswith("--lcpp-gguf=")):
            LLAMA_ConvertToGGUF = arg[12:]
        elif (arg.startswith("--lcpp-pre-quant=")):
            LLAMA_PreQuantize = arg[17:]
        elif (arg.startswith("--lcpp-quant=")):
            LLAMA_Quantize = arg[13:]
        elif (arg.startswith("--model-card-template=")):
            LLM_ModelCardTemplate = arg[22:]
        elif (arg.startswith("--repo-name-template=")):
            LLM_CreateRepoNameTemplate = arg[21:]
        elif (arg == "--repo-public"):
            LLM_CreateRepoPrivate = False
        elif (arg == "--test"):
            TestingMode = True
        else:
            logging.Log(logging.CRIT, f"Unknown argument '{arg}'.")
            raise ValueError()

if (not os.path.exists(f"{LLM_CacheDir}/")):
    logging.Log(logging.LOG_LEVEL_INFO, "Cache directory doesn't exists. Creating.")

if (not os.path.exists(f"{GGUF_Output}/")):
    logging.Log(logging.LOG_LEVEL_INFO, "Output directory doesn't exists. Creating.")
    os.mkdir(GGUF_Output)

if (not os.path.exists(f"{GGUF_Output}/{GetRepoName()}/")):
    logging.Log(logging.LOG_LEVEL_INFO, "Model output directory doesn't exists. Creating.")

    if (not TestingMode):
        os.mkdir(f"{GGUF_Output}/{GetRepoName()}")

if (len(Quants) == 0):
    logging.Log(logging.LOG_LEVEL_WARN, "Quants list is empty, converting only to GGUF.")

if (LLM_Repo is None or not RepoExists(LLM_Repo)):
    logging.Log(logging.LOG_LEVEL_CRIT, "Model repo is null or doesn't exists. Not able to continue.")
    raise RuntimeError()

if (GGUF_File is None):
    if (not os.path.exists(f"{LLM_CacheDir}/{LLM_Repo.replace('/', '_')}/")):
        logging.Log(logging.LOG_LEVEL_INFO, "Model doesn't exists in cache. Downloading.")
        
        if (not TestingMode):
            snapshot_download(repo_id = LLM_Repo, local_dir = f"{LLM_CacheDir}/{LLM_Repo.replace('/', '_')}/", local_dir_use_symlinks = False)

    logging.Log(logging.LOG_LEVEL_INFO, "Converting model to GGUF.")

    outtype = GetQuantInfo(GGUF_OutType, True)
    outfile = f"{GGUF_Output}/{GetRepoName()}/{LLM_Repo.replace('/', '_')}.gguf"

    if ("lcpp_name" not in outtype or outtype["lcpp_name"] is None):
        outtype = GGUF_OutType.upper()
    else:
        outtype = outtype["lcpp_name"].upper()

    if (outtype != "F32" and outtype != "BF16" and outtype != "F16" and outtype != "Q8_0"):
        logging.Log(logging.LOG_LEVEL_CRIT, "Invalid output type. Must be F32, BF16, F16, or Q8_0.")
        raise RuntimeError()

    if (not TestingMode):
        ExecuteCommand((f"'{LLAMA_PreConvertToGGUF}' " if (len(LLAMA_PreConvertToGGUF) > 0) else "") + f"{LLAMA_Dir}/{LLAMA_ConvertToGGUF}' '{LLM_CacheDir}/{LLM_Repo.replace('/', '_')}' --outtype {outtype.lower()} --outfile '{outfile}'", True)

    GGUF_File = outfile

    logging.Log(logging.LOG_LEVEL_INFO, "Model converted to GGUF!")
    logging.Log(logging.LOG_LEVEL_INFO, "Removing cache dirrectory.")

    if (not TestingMode):
        shutil.rmtree(f"{LLM_CacheDir}/{LLM_Repo.replace('/', '_')}")

modelCardTable = "|Quant|Size|Description|\n|---|---|---|\n"

for quant in Quants:
    if (quant == GetQuantInfo(GGUF_OutType, True)):
        logging.Log(logging.LOG_LEVEL_ERRO, f"Quant type ({quant}) is the same as the GGUF out type. Skipping.")
        continue

    logging.Log(logging.LOG_LEVEL_INFO, f"Quantizing model to '{quant}'.")

    quantInfo = GetQuantInfo(quant, True)
    outfile = f"{GGUF_Output}/{GetRepoName()}/{LLM_Repo.replace('/', '_')}_{quant}.gguf"

    if ("lcpp_name" not in quantInfo or quantInfo["lcpp_name"] is None):
        quantName = quant
    else:
        quantName = quantInfo["lcpp_name"]

    if ("extra_params" not in quantInfo or quantInfo["extra_params"] is None):
        quantExtraParams = ""
    else:
        quantExtraParams = quantInfo["extra_params"]

    if ("description" not in quantInfo or quantInfo["description"] is None):
        quantDescription = ""
    else:
        quantDescription = quantInfo["description"]

    if (not TestingMode):
        ExecuteCommand((f"'{LLAMA_PreQuantize}' " if (len(LLAMA_PreQuantize) > 0) else "") + f"'{LLAMA_Dir}/{LLAMA_Quantize}' {quantExtraParams} '{GGUF_File}' '{outfile}' {quantName}", True)

        modelSize = os.path.getsize(outfile)
        modelSize = FormatSize(modelSize)
        modelSize = f"{modelSize[0]} {modelSize[1]}"
    else:
        modelSize = "00 B"

    modelCardTable += f"|[{quant}](https://huggingface.co/{HF_Username}/{GetRepoName()}/resolve/main/{outfile})|{modelSize}|{quantDescription}|\n"
    logging.Log(logging.LOG_LEVEL_INFO, f"Model quantized to '{quant}'!")

if (not TestingMode):
    originalModelSize = os.path.getsize(GGUF_File)
    originalModelSize = FormatSize(originalModelSize)
    originalModelSize = f"{originalModelSize[0]} {originalModelSize[1]}"
else:
    originalModelSize = "00 B"

originalModelOTInfo = GetQuantInfo(GGUF_OutType, True)

if ("description" not in originalModelOTInfo or originalModelOTInfo["description"] is None):
    originalModelOTDescription = ""
else:
    originalModelOTDescription = originalModelOTInfo["description"]

modelCardTable += f"|[ORIGINAL ({GGUF_OutType})](https://huggingface.co/{HF_Username}/{GetRepoName()}/resolve/main/{GGUF_File[GGUF_File.rfind('/') + 1:] if ('/' in GGUF_File) else GGUF_File})|{originalModelSize}|{originalModelOTDescription}|"

modelCardMetadata = f"---\nbase_model:\n- {LLM_Repo}\npipeline_tag: text-generation\n---\n"
modelCard = LLM_ModelCardTemplate.replace("[METADATA]", modelCardMetadata).replace("[TABLE]", modelCardTable).replace("[IN_REPO]", LLM_Repo).replace("[OUT_REPO]", f"{HF_Username}/{GetRepoName()}")

if (not TestingMode):
    with open(f"{GGUF_Output}/{GetRepoName()}/README.md", "w+") as f:
        f.write(modelCard)

logging.Log(logging.LOG_LEVEL_INFO, "All quantizations completed!")
logging.Log(logging.LOG_LEVEL_INFO, f"Model card:\n```markdown\n{modelCard}\n```")

if (not RepoExists(f"{HF_Username}/{GetRepoName()}")):
    logging.Log(logging.LOG_LEVEL_INFO, "Output repository doesn't exists. Creating.")

    if (not TestingMode):
        create_repo(f"{HF_Username}/{GetRepoName()}", private = LLM_CreateRepoPrivate)

logging.Log(logging.LOG_LEVEL_INFO, f"Uploading contents of '{GGUF_Output}/{GetRepoName()}' (local) to '{HF_Username}/{GetRepoName()}' (remote).")

if (not TestingMode):
    HF_API.upload_folder(folder_path = f"{GGUF_Output}/{GetRepoName()}", repo_id = f"{HF_Username}/{GetRepoName()}")

logging.Log(logging.LOG_LEVEL_INFO, "Removing output directory.")

if (not TestingMode):
    shutil.rmtree(f"{GGUF_Output}/{GetRepoName()}")

logging.Log(logging.LOG_LEVEL_INFO, f"Everything is completed. Check out https://huggingface.co/{HF_Username}/{GetRepoName()}")
