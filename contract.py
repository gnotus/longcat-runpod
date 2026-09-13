"""Cheap validation happens before downloads or GPU model initialization."""
import base64
import math

def validate(data, mode):
    if mode not in ('sf', 'avatar'):
        raise ValueError('LONGCAT_MODE must be sf or avatar')
    allowed = {'prompt','start_frame','audio','duration','resolution','seed'}
    if set(data) - allowed:
        raise ValueError('Unsupported fields: ' + ', '.join(sorted(set(data)-allowed)))
    if not isinstance(data.get('prompt'), str) or not data['prompt'].strip():
        raise ValueError('Nonempty prompt required')
    if len(data['prompt']) > 12000:
        raise ValueError('Prompt too long')
    if not str(data.get('start_frame','')).startswith('data:image/'):
        raise ValueError('Original start_frame required as an image data URI')
    if data.get('resolution','720p') not in ('480p','720p'):
        raise ValueError('Resolution must be 480p or 720p')
    duration = float(data.get('duration',15))
    if not math.isfinite(duration) or not 1 <= duration <= 15:
        raise ValueError('Duration must be 1..15 seconds')
    if mode == 'avatar' and not str(data.get('audio','')).startswith('data:audio/'):
        raise ValueError('Avatar requires supplied speech as an audio data URI')
    if mode == 'sf' and data.get('audio'):
        raise ValueError('SF route generates silent video; audio is not supported')
    fps = 25 if mode == 'avatar' else 15
    frames = 4 * math.ceil((duration*fps-1)/4) + 1
    return {**data, 'duration':duration,'fps':fps,'num_frames':frames,
            'resolution':data.get('resolution','720p'),'seed':int(data.get('seed',4715))}

def decode_uri(uri, kind, max_bytes):
    header, sep, body = uri.partition(',')
    if not sep or not header.startswith('data:'+kind+'/') or not header.endswith(';base64'):
        raise ValueError('Invalid base64 data URI')
    raw = base64.b64decode(body, validate=True)
    if not raw or len(raw) > max_bytes:
        raise ValueError('Attachment size out of range')
    return raw
