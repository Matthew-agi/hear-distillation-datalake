# Third-party notices

`src/hear_distill/audio.py` is derived from the Google Health HeAR audio
preprocessing implementation at
<https://github.com/Google-Health/hear/blob/master/python/data_processing/audio_utils.py>.
It retains the original copyright notice and is distributed under the Apache
License 2.0. The implementation in this repository caches invariant tensors
and vectorizes the PCEN exponential moving average.
