import argparse
import os
import subprocess
import sysconfig
from pathlib import Path

import mlx
import nanobind

r = Path(__file__).parent
m = Path(mlx.__path__[0])
# A distribution whose MLX is not a pip layout (headers and libmlx in a prefix of
# their own, as MacPorts installs them) names that prefix and the compiler here.
# Without the variables the build is unchanged.
prefix = os.environ.get("MLX_PREFIX")
include = Path(prefix) / "include" if prefix else m / "include"
lib = Path(prefix) / "lib" if prefix else m / "lib"
rpath = str(lib) if prefix else "@loader_path/../mlx/lib"
n = Path(nanobind.__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=r.parent / "src/rec_mlx")
a = parser.parse_args()
out = a.output / ("_grammar_native" + sysconfig.get_config_var("EXT_SUFFIX"))
subprocess.run(
    [
        os.environ.get("CXX", "clang++"),
        "-std=c++20",
        "-DNB_DOMAIN=mlx",
        "-O3",
        "-shared",
        "-fPIC",
        "-fvisibility=hidden",
        "-undefined",
        "dynamic_lookup",
        "-Wl,-rpath," + rpath,
        "-I" + str(include),
        "-I" + str(n / "include"),
        "-I" + str(n / "ext/robin_map/include"),
        "-I" + sysconfig.get_path("include"),
        str(r / "grammar_native.cpp"),
        str(n / "src/nb_combined.cpp"),
        "-L" + str(lib),
        "-lmlx",
        "-o",
        str(out),
    ],
    check=True,
)
print(out)
