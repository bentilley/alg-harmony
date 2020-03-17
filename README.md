# Algorithmic Harmony

This project generates an evolving harmony in real time.

This repo contains a graph written in TensorFlow v1 (recently ported to v2
using `tf.compat`). The graph was loaded by an audio plugin written in C++
(using the JUCE framework) and run to generate an improvised polyphony.

## Project Structure

The meat of the project is written in the `harmony/harmony.py` file. Running
this compiles a graph and exports the resulting `pb` file to `/opt/alg-harmony`
(from which it was loaded by the C++ code). You need to make an
`/opt/alg-harmony` symlink to somewhere writeable, otherwise you will get a
permission error.

## Algorithm

The algorithm implemented used swam intelligence to decide what notes to play
based on what was currently being played. Unplayed notes were assigned a value
based on their coherence with the current harmony and attracted more energy,
increasing their volume.
