from Bio import SeqIO
from Bio.PDB.Polypeptide import d1_to_index
import numpy as np

def kmers(seq, n):
    kpos = []
    for i in range(0, len(seq)-n):
        kmer = seq[i:(i+n)]
        kpos.append(min(kmer, kmer[::-1]))
    return kpos

def aaindex(seq):
    ix = []
    for a in seq:
        if a in d1_to_index:
            ix.append( d1_to_index[a] )
    return ix


def readfasta(ffile):
    record_dict = SeqIO.to_dict(SeqIO.parse(ffile, "fasta"))
    return record_dict

def train_test_split(n, r=0.10, random=True):
    ix = np.arange(n)
    if random:
        ix = np.random.shuffle(ix)
    split = np.ceil( n*(1-r) )
    train = ix[0:split]
    test = ix[split:n]
    return train, test

def dbfasta(fasfile):
    seqdict = {}
    for seq in SeqIO.parse(fasfile, "fasta"):
        seqid = seq.id
        info = seqid.split('|')
        if len(info) >= 2:
            uniprot = info[1]
        seqdict[uniprot] = seq
    return seqdict

def seqinfo(infofile):
    seqinfo = {}
    ecinfo = {}
    for line in open(infofile):
        m = line.rstrip().split('\t')
        if m[1] == 'uniprot':
            rid = m[0]
            uniprot = m[2]
            rinfo = m[3].split('|')
            try:
                ec = m[4].split('|')
                if uniprot not in seqinfo:
                    seqinfo[uniprot] = []
                seqinfo[uniprot].append( (rid, rinfo, ec) )
                for e in ec:
                    if e not in ecinfo:
                        ecinfo[e] = set()
                    ecinfo[e].add(uniprot)
            except:
                continue
    return seqinfo, ecinfo

def ecdataset(seqdict, ecinfo):
    data = {}
    for ec in ecinfo:
        if ec not in data:
            data[ec] = []
        for seqid in ecinfo[ec]:
            if seqid in seqdict:
                seq = str(seqdict[seqid].seq)
                data[ec].append( (seqid, seq) )
    return data

def printecinfo(data):
    for ec in sorted(data):
        print (ec, len(data[ec]))

def label2class(llist):
    cl = sorted( set(llist) )
    cln = [cl.index(w) for w in llist]
    return cln, cl

def dataset(data, eclist, minsize=100, shuffle=True):
    seqs = []
    seqids = []
    ectrain = []
    for ec in sorted(eclist):
        if ec in data and len(data[ec]) >= minsize:
            if shuffle:
                np.random.shuffle(data[ec])
            for i in range(0, minsize):
                sinfo = data[ec][i]
                seqids.append( sinfo[0] )
                seqs.append( sinfo[1] )
                ectrain.append(ec)
    cl, clids = label2class(ectrain)
    return seqs, seqids, cl, ectrain
