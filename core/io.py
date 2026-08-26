#!/usr/bin/env python
#_*_coding:utf-8_*_

from Bio import SeqIO

def read_fasta(filename):

    with open(filename) as f:
        records = list(SeqIO.parse(f, "fasta"))

    return [(r.id, str(r.seq)) for r in records]
