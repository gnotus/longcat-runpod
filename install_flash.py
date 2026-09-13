import subprocess
import sys
import torch
version = f'cp{sys.version_info.major}{sys.version_info.minor}'
abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
name=f'flash_attn-2.7.4.post1+cu12torch2.6cxx11abi{abi}-{version}-{version}-linux_x86_64.whl'
subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir',
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/'+name],check=True)
