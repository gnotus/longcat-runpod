import torch
import flash_attn
import worker
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
assert torch.__version__.startswith('2.6.0')
assert flash_attn.__version__ == '2.7.4.post1'
print('LONGCAT_IMAGE_IMPORTS_PASS',torch.__version__,torch.version.cuda)
