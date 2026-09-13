# LongCat SF and Avatar 1.5 RunPod worker

Separate `LONGCAT_MODE=sf` and `LONGCAT_MODE=avatar` endpoints, one H200 each,
minimum zero/maximum one worker. SF uses the 16-step CFG/step adapter; Avatar
uses the mandatory eight-step DMD adapter. Models persist across requests.

Input: prompt, start_frame (original image data URI), duration (1..15s),
resolution (480p or 720p), seed. Avatar additionally requires supplied speech
as an audio data URI. End-frame fields are rejected, never silently ignored.
Avatar is not a voice-cloning/TTS service. Speech should be clean vocals.

Output is a verified encoded MP4 in configured S3 storage, or small inline MP4.
No character assets, audio, account credentials or production prompts belong
in this repository. GPU validation and quality acceptance remain required.

Upstream: https://github.com/meituan-longcat/LongCat-Video (MIT).
Pinned source and model revisions are recorded in Dockerfile and worker.py.
