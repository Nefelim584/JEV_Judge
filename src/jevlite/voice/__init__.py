"""Voice pipeline: live speech-to-text whose final transcript per utterance goes to jev-lite (todo Phase 9b).

- ``endpoint``: end-of-utterance detection (Silero VAD silence, optionally Smart Turn v3.2).
- ``audio``: test streams built from public speech (utterance + silence + next utterance, so the end of
  speech, t0, is known exactly).
- ``endpoint_study``: t1 − t0 (endpoint wait) and false endpoints per endpoint setting, on the CPU.
"""
