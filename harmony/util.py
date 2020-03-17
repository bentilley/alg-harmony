
def get_midi_note_in_hertz(note_number, frequency_of_a=440):
    return frequency_of_a * pow(2.0, (note_number - 69) / 12.0)

class LinearAttractionFromLog():

    # Given pitch positions of attractor and attractee, return linear attraction relative
    # to range_in_hertz

    def __init__(self, range_in_hz=15, exclude_same_index=False):
        self.range = range_in_hz
        self.exclude_same_index = exclude_same_index

    def __call__(self, attractor, attractee):
        attractor_freq = get_midi_note_in_hertz(attractor)
        attractee_freq = get_midi_note_in_hertz(attractee)
        freq_diff = abs(attractor_freq - attractee_freq)
        if freq_diff > self.range:
            return 0
        if (freq_diff == 0):
            if self.exclude_same_index:
                return 0
            else:
                return 1
        else:
            return self.range / freq_diff
