INPUT_SAMPLE_RATE = 16_000
TARGET_SAMPLE_RATE = 49.95  # 50 (16000/320) wasnt exactly correct
WAV2VEC_FRAME_LEN = 20 # length of a wav2vec 2.0 model predicitons (in ms)
HIDDEN_SIZE = 1024  # output dimensionality of wav2vec 2.0 models with 300m params
NOISE_THRESHOLD = 0.1  # the duration (in seconds), below which a segment is considered to be noise and is thus excluded