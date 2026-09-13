import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from contract import validate, decode_uri
base={'prompt':'A character waves','start_frame':'data:image/png;base64,YQ==','duration':15}
assert validate(base,'sf')['num_frames']==225
assert validate({**base,'audio':'data:audio/wav;base64,YQ=='},'avatar')['num_frames']==377
for data,mode in [({**base,'end_frame':'x'},'sf'),({**base,'duration':float('nan')},'sf'),
                  ({**base,'start_frame':''},'sf'),(base,'avatar'),({**base,'audio':'x'},'sf')]:
    try: validate(data,mode)
    except (ValueError,TypeError): pass
    else: raise AssertionError('Invalid input accepted')
assert decode_uri('data:image/png;base64,YQ==','image',1)==b'a'
print('LONGCAT_CONTRACT_PASS')
