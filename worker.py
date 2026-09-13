"""One loaded pipeline per worker; native SF or Avatar 1.5, never fake EF."""
import base64
import io
import json
import os
import pathlib
import subprocess
import tempfile
import threading
import time
import uuid

from contract import validate, decode_uri

MODE = os.environ.get('LONGCAT_MODE','sf')
PIPE = None
LOCK = threading.Lock()
VIDEO_REV = '03b55529b1d1d4045f5fbe14d65c8c6e8116b278'
AVATAR_REV = '92016c71d5d318d0f5d84e4db30015a571484ab6'

def load_pipeline():
    global PIPE
    if PIPE is not None:
        return PIPE
    import torch
    import torch.distributed as dist
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, UMT5EncoderModel
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.context_parallel.context_parallel_util import init_context_parallel, get_optimal_split
    assert torch.cuda.device_count() == 1, 'Exactly one CUDA GPU required'
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        rendezvous = pathlib.Path(tempfile.mkdtemp(prefix='longcat-dist-'))/'rendezvous'
        dist.init_process_group('nccl', init_method='file://'+str(rendezvous),rank=0,world_size=1)
    init_context_parallel(context_parallel_size=1,global_rank=0,world_size=1)
    split = get_optimal_split(1)
    patterns = ['tokenizer/*','text_encoder/*','vae/*','scheduler/*']
    if MODE == 'sf':
        patterns += ['dit/*','lora/cfg_step_lora.safetensors']
    print(json.dumps({'stage':'weights','mode':MODE}),flush=True)
    video = pathlib.Path(snapshot_download('meituan-longcat/LongCat-Video',revision=VIDEO_REV,allow_patterns=patterns))
    common = dict(tokenizer=AutoTokenizer.from_pretrained(video,subfolder='tokenizer'),
        text_encoder=UMT5EncoderModel.from_pretrained(video,subfolder='text_encoder',torch_dtype=torch.bfloat16),
        vae=AutoencoderKLWan.from_pretrained(video,subfolder='vae',torch_dtype=torch.bfloat16))
    if hasattr(common['vae'],'enable_tiling'):
        common['vae'].enable_tiling()
    if MODE == 'sf':
        from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
        dit = LongCatVideoTransformer3DModel.from_pretrained(video,subfolder='dit',cp_split_hw=split,torch_dtype=torch.bfloat16)
        lora = video/'lora/cfg_step_lora.safetensors'
        assert lora.is_file(), 'Missing SF distilled adapter'
        dit.load_lora(str(lora),'cfg_step_lora')
        dit.enable_loras(['cfg_step_lora'])
        PIPE = LongCatVideoPipeline(**common,dit=dit,scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(video,subfolder='scheduler'))
    else:
        from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
        from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
        from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
        avatar = pathlib.Path(snapshot_download('meituan-longcat/LongCat-Video-Avatar-1.5',revision=AVATAR_REV,
            allow_patterns=['base_model/*','scheduler/*','lora/dmd_lora.safetensors','whisper-large-v3/*']))
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(avatar,subfolder='base_model',cp_split_hw=split,torch_dtype=torch.bfloat16)
        lora = avatar/'lora/dmd_lora.safetensors'
        assert lora.is_file(), 'Missing Avatar 1.5 DMD adapter'
        dit.load_lora(str(lora),'dmd',multiplier=1.0,lora_network_dim=128,lora_network_alpha=64)
        dit.enable_loras(['dmd'])
        audio_dir = str(avatar/'whisper-large-v3')
        PIPE = LongCatVideoAvatarPipeline(**common,dit=dit,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(avatar,subfolder='scheduler'),
            audio_encoder=get_audio_encoder(audio_dir,'avatar-v1.5').to('cuda:0'),
            audio_feature_extractor=get_audio_feature_extractor(audio_dir,'avatar-v1.5'),model_type='avatar-v1.5')
    PIPE.to('cuda:0')
    print(json.dumps({'stage':'model_ready','mode':MODE,'steps':8 if MODE=='avatar' else 16,
                      'gpu':torch.cuda.get_device_name(0)}),flush=True)
    return PIPE

def render(job):
    data = validate(job['input'],MODE)
    from PIL import Image
    image = Image.open(io.BytesIO(decode_uri(data['start_frame'],'image',6_000_000)))
    image.load()
    image = image.convert('RGB')
    audio_bytes = decode_uri(data['audio'],'audio',3_000_000) if MODE=='avatar' else None
    started = time.monotonic()
    pipe = load_pipeline()
    load_seconds = time.monotonic()-started
    import torch
    import numpy as np
    import imageio.v2 as imageio
    generated = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='longcat-output-') as temporary:
        root = pathlib.Path(temporary)
        kwargs = dict(image=image,prompt=data['prompt'],resolution=data['resolution'],
            num_frames=data['num_frames'],generator=torch.Generator(device='cuda:0').manual_seed(data['seed']),
            use_distill=True)
        with torch.inference_mode():
            if MODE=='sf':
                frames = pipe.generate_i2v(**kwargs,num_inference_steps=16,guidance_scale=1.0)[0]
            else:
                import librosa
                audio_path = root/'input_audio.wav'
                audio_path.write_bytes(audio_bytes)
                speech,sr = librosa.load(audio_path,sr=16000)
                target_samples = int(np.ceil(data['num_frames']/data['fps']*sr))
                speech = np.pad(speech[:target_samples],(0,max(0,target_samples-len(speech))))
                full = pipe.get_audio_embedding(speech,fps=25,device='cuda:0',sample_rate=sr,model_type='avatar-v1.5')
                if not torch.isfinite(full).all():
                    raise ValueError('Nonfinite audio embedding')
                centers = torch.arange(data['num_frames'])[:,None] + (torch.arange(5)-2)[None,:]
                centers = centers.clamp(0,full.shape[0]-1)
                embedding = full[centers][None,...].to('cuda:0')
                out,_ = pipe.generate_ai2v(**kwargs,num_inference_steps=8,text_guidance_scale=1.0,
                    audio_guidance_scale=1.0,audio_emb=embedding,output_type='both',resize_mode='default')
                frames = out[0]
        generation_seconds = time.monotonic()-generated
        raw_video = root/'silent.mp4'
        with imageio.get_writer(raw_video,fps=data['fps'],codec='libx264',quality=8,macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(np.clip(np.asarray(frame)*255,0,255).astype(np.uint8))
        result = raw_video
        if MODE=='avatar':
            result = root/'avatar.mp4'
            subprocess.run(['ffmpeg','-v','error','-i',str(raw_video),'-i',str(audio_path),
                '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-af','apad',
                '-t',str(data['num_frames']/data['fps']),str(result)],check=True)
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(result)]))
        subprocess.run(['ffmpeg','-v','error','-i',str(result),'-f','null','-'],check=True)
        stream = next(s for s in probe['streams'] if s['codec_type']=='video')
        duration = float(probe['format']['duration'])
        assert abs(duration-data['num_frames']/data['fps'])<0.2, 'Output duration mismatch'
        info = dict(mode=MODE,requested_seconds=data['duration'],duration_seconds=duration,
            fps=data['fps'],width=stream['width'],height=stream['height'],seed=data['seed'],
            model_load_seconds=round(load_seconds,3),generation_seconds=round(generation_seconds,3),
            steps=8 if MODE=='avatar' else 16,has_audio=MODE=='avatar',gpu=torch.cuda.get_device_name(0))
        if os.environ.get('SPACES_BUCKET'):
            import boto3
            key = 'longcat-tests/'+MODE+'/'+str(uuid.uuid4())+'.mp4'
            s3 = boto3.client('s3',endpoint_url=os.environ['SPACES_ENDPOINT'],
                aws_access_key_id=os.environ['SPACES_ACCESS_KEY'],aws_secret_access_key=os.environ['SPACES_SECRET_KEY'],
                region_name=os.environ['SPACES_REGION'])
            s3.upload_file(str(result),os.environ['SPACES_BUCKET'],key,ExtraArgs={'ContentType':'video/mp4'})
            info.update(s3_uri='s3://'+os.environ['SPACES_BUCKET']+'/'+key)
        else:
            assert result.stat().st_size<6_500_000,'Encoded result needs object storage'
            info['mp4_base64']=base64.b64encode(result.read_bytes()).decode()
        return info

def handler(job):
    with LOCK:
        return render(job)

if __name__=='__main__':
    import runpod
    runpod.serverless.start({'handler':handler})
