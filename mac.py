import ctypes
import ctypes.util
import os
import platform
import subprocess
import sys

THERMAL = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
_thermal_fn = None
_thermal_failed = False


def _sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def info() -> dict:
    if sys.platform != "darwin":
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        return {"chip": platform.machine(), "model": platform.system(), "ram_gb": round(ram), "fanless": False}
    chip = _sh("sysctl", "-n", "machdep.cpu.brand_string")
    ram = int(_sh("sysctl", "-n", "hw.memsize")) / 2**30
    model = next((l.split(":", 1)[1].strip() for l in _sh("system_profiler", "SPHardwareDataType").splitlines()
                  if "Model Name" in l), "Mac")
    # every MacBook Air (and the 12" MacBook) has no fan, so it throttles on long runs
    return {"chip": chip, "model": model, "ram_gb": round(ram), "fanless": "Air" in model}


def _load_thermal():
    # NSProcessInfo.thermalState through the objc runtime, so no pyobjc needed
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.restype = ctypes.c_void_p
    addr = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    send_ptr = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(addr)
    send_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(addr)
    proc = send_ptr(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
    sel = objc.sel_registerName(b"thermalState")
    return lambda: send_long(proc, sel)


def thermal() -> int | None:
    """0 nominal, 1 fair, 2 serious, 3 critical. None off macOS."""
    global _thermal_fn, _thermal_failed
    if sys.platform != "darwin" or _thermal_failed:
        return None
    if _thermal_fn is None:
        try:
            _thermal_fn = _load_thermal()
        except (OSError, AttributeError, TypeError) as e:
            print(f"warning: can't read thermal state ({e}), training continues without it")
            _thermal_failed = True
            return None
    return int(_thermal_fn())
