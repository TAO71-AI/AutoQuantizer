from enum import Enum
from datetime import datetime as DT

LOG_LEVEL_INFO: tuple[str, int] = ("INFO", 0)
LOG_LEVEL_WARN: tuple[str, int] = ("WARN", 1)
LOG_LEVEL_ERRO: tuple[str, int] = ("ERRO", 2)
LOG_LEVEL_CRIT: tuple[str, int] = ("CRIT", 3)

def Log(
    Level: str | int | tuple[str, int],
    Message: str,
    Start: str = "",
    End: str = "\n",
    Flush: bool = True,
    PrintDate: bool = True
) -> None:
    if (Level == LOG_LEVEL_INFO[0] or Level == LOG_LEVEL_INFO[1] or Level == LOG_LEVEL_INFO):
        level = LOG_LEVEL_INFO
    elif (Level == LOG_LEVEL_WARN[0] or Level == LOG_LEVEL_WARN[1] or Level == LOG_LEVEL_WARN):
        level = LOG_LEVEL_WARN
    elif (Level == LOG_LEVEL_ERRO[0] or Level == LOG_LEVEL_ERRO[1] or Level == LOG_LEVEL_ERRO):
        level = LOG_LEVEL_ERRO
    elif (Level == LOG_LEVEL_CRIT[0] or Level == LOG_LEVEL_CRIT[1] or Level == LOG_LEVEL_CRIT):
        level = LOG_LEVEL_CRIT
    else:
        raise ValueError("Unknown log level.")

    date = DT.now()

    print((f"[{date.hour}:{date.minute}:{date.second}] " if (PrintDate) else "") + f"[{level[0]}] {Start}{Message}{End}", end = "", flush = Flush)

if (__name__ == "__main__"):
    Log(LOG_LEVEL_INFO, "Test 1")
    Log(LOG_LEVEL_WARN, "Test 2")
    Log(LOG_LEVEL_ERRO, "Test 3")
    Log(LOG_LEVEL_CRIT, "Test 4")
