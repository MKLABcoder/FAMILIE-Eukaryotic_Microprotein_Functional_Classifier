#!/usr/bin/env python
#_*_coding:utf-8_*_

import re
from collections import Counter

def AAC(fastas, order=None):
    """
    Amino Acid Composition
    fastas: list of (id, sequence)
    order: amino acid order string
    """
    if order is None:
        order = 'ACDEFGHIKLMNPQRSTVWY'

    encodings = []
    header = ['#'] + list(order)
    encodings.append(header)

    for name, seq in fastas:
        sequence = re.sub('-', '', seq)
        count = Counter(sequence)
        code = [name]
        for aa in order:
            code.append(count.get(aa, 0) / len(sequence))
        encodings.append(code)

    return encodings

