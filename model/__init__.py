"""SEAL — reward-based e-prop LSNN on ALE Pong.

Single backend: recurrent network of spiking neurons (LIF + ALIF) trained
online by adaptive e-prop (Bellec et al., Nature Communications 2020).
No BPTT, no replay, no frame stacking. One frame in -> one action out ->
one e-prop update per env step.
"""
