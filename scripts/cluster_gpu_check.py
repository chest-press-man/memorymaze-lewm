import importlib.metadata
import json
from pathlib import Path
import platform

import torch

assert torch.cuda.is_available(), 'CUDA is not available in the allocated Job'
x=torch.ones((256,256),device='cuda');assert (x@x)[0,0].item()==256
report={'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'cuda_matmul_passed':True,'device_scope':'allocated cloud Job, not login node'}
out=Path('execution');out.mkdir(exist_ok=True)
(out/'cloud_environment.json').write_text(json.dumps(report,indent=2)+'\n')
packages=sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
(out/'environment.lock.txt').write_text('\n'.join(packages)+'\n')
print(json.dumps(report),flush=True)
