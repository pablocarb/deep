""" Tools for deep learning encoding """
from Bio import SeqIO
from Bio.PDB.Polypeptide import d1_to_index
import numpy as np
import os,glob
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.rdmolops import RDKFingerprint
import csv
import sys


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

def _dbfasta(fasfile):
    seqdict = {}
    for seq in SeqIO.parse(fasfile, "fasta"):
        seqid = seq.id
        info = seqid.split('|')
        if len(info) >= 2:
            uniprot = info[1]
        seqdict[uniprot] = seq
    return seqdict

def _seqinfo(infofile):
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

def _ecdataset(seqdict, ecinfo):
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

def ecdataset(TEST=False):
    fasfile = os.path.join('/mnt/SBC1/data/METANETX2', 'seqs.fasta')
    infofile = os.path.join('/mnt/SBC1/data/METANETX2', 'reac_seqs.tsv')

    seqdict = tools._dbfasta(fasfile)
    seqinfo, ecinfo = tools._seqinfo(infofile)
    data = tools._ecdataset(seqdict, ecinfo)

    pattern = ''
    eclist = set()
    for ec in ecinfo:
        if ec.startswith(pattern):
            eclist.add(ec)
    if TEST:
        ectest = set(['1.4.1.13','2.2.1.2','3.1.1.3','4.1.1.11','5.1.1.1','6.1.1.18'])
        eclist = ectest

    seqs, seqids, Y, Yids = tools.dataset(data, eclist)
    return seqs, seqids, Y, Yids

def thermodataset(balanced=False):
    folder = '/mnt/SBC1/data/thermostability/montanucci08'

    ffile = os.path.join(folder, 'Allsequences.fasta')
    rd = SeqIO.to_dict(SeqIO.parse(ffile, "fasta"))
    """ doi:  10.1093/bioinformatics/btn166 """
    """ Left: thermophilic microbial organism, right: a mesophilic one. """
    seqs = []
    seqids = []
    Y = []
    for clus in glob.glob(os.path.join(folder, 'cluster.*')):
        for line in open(clus):
            left, right = line.rstrip().split()
            s1 = str(rd[left].seq)
            s2 = str(rd[right].seq)
            if not balanced:
                if s1 not in seqs:
                    seqs.append(s1)
                    seqids.append(left)
                    Y.append(1)
                if s2 not in seqs:
                    seqs.append(s2)
                    seqids.append(right)
                    Y.append(0)
            else:
                if s1 not in seqs and s2 not in seqs:
                    seqs.append(s1)
                    seqids.append(left)
                    Y.append(1)
                    seqs.append(s2)
                    seqids.append(left)
                    Y.append(0)
    cl, clids = label2class(Y)
    return seqs, seqids, cl, Y

def thermodataset2():
    folder = '/mnt/SBC1/data/thermostability/lin10'
    """ doi:  10.1016/j.mimet.2010.10.0131 """
    """ h.txt: thermophilic, l.txt: mesophilic. """
    seqs = []
    seqids = []
    Y = []
    f1 = os.path.join(folder, 'l.txt')
    rd1 = SeqIO.to_dict(SeqIO.parse(f1, "fasta"))
    for s in sorted(rd1):
        seqs.append( str(rd1[s]) )
        seqids.append( s )
        Y.append( 0 )
    f1 = os.path.join(folder, 'h.txt')
    rd1 = SeqIO.to_dict(SeqIO.parse(f1, "fasta"))
    for s in sorted(rd1):
        seqs.append( str(rd1[s]) )
        seqids.append( s )
        Y.append( 1 )

    cl, clids = label2class(Y)
    return seqs, seqids, cl, Y

def reacDataset():
    rset = []
    csv.field_size_limit(sys.maxsize) # To avoid error with long csv fields
    rsmiFile = os.path.join('/mnt/SBC1/data/METANETX2', 'reac_smi.csv')
    with open(rsmiFile) as f:
        for row in csv.DictReader(f):
            rid = row['RID']
            if rlist is not None
            smi = row['SMILES']
            left, right = smi.split('>>')
            rleft = left.split('.')
            rright = right.split('.')
            ok = True
            mleft = []
            mright = []
            for c in rleft:
                if c not in smiles:
                    try:
                        smiles[c] = MolFromSmiles(c)
                    except:
                        ok = False
                        break
                mleft.append((c, smiles[c]))
            if not ok:
                continue
            for c in rright:
                if c not in smiles:
                    try:
                        smiles[c] = MolFromSmiles(c)
                    except:
                        ok = False
                        break
                mright.append((c, smiles[c]))
            if not ok:
                continue
            rset.append( (rid, mleft, mright, rleft, rright) )
    return rset

def reactionFingerprint(radius=5, rlist=None):
    """ Reaction binary fingerprint based on prod-subs fingerprint logic difference """
    """ Suitable for training sets or output sets """
    """ We use RDKit fingerprints with selected radius """
    """ rsmifile: precomputed reaction SMILES from METANETX2 """
    csv.field_size_limit(sys.maxsize) # To avoid error with long csv fields
    rsmiFile = os.path.join('/mnt/SBC1/data/METANETX2', 'reac_smi.csv')
    smiles = {}
    fps = {}
    rfp = {}
    with open(rsmiFile) as f:
        for row in csv.DictReader(f):
            rid = row['RID']
            if rlist is not None
            smi = row['SMILES']
            left, right = smi.split('>>')
            rleft = left.split('.')
            rright = right.split('.')
            ok = True
            mleft = []
            mright = []
            for c in rleft:
                if c not in smiles:
                    try:
                        smiles[c] = MolFromSmiles(c)
                    except:
                        ok = False
                        break
                mleft.append((c, smiles[c]))
            if not ok:
                continue
            for c in rright:
                if c not in smiles:
                    try:
                        smiles[c] = MolFromSmiles(c)
                    except:
                        ok = False
                        break
                mright.append((c, smiles[c]))
            if not ok:
                continue
            for c in mright:
                if c[0] not in fps:
                    try:
                        fps[c[0]] = RDKFingerprint(c[1], minPath=1, maxPath=radius)
                    except:
                        ok = False
                        break
                if rid not in rfp:
                    rfp[rid] = fps[c[0]]
                else:
                    rfp[rid] = rfp[rid] | fps[c[0]]
            if not ok:
                del rfp[rid]
            for c in mleft:
                if c[0] not in fps:
                    try:
                        fps[c[0]] = RDKFingerprint(c[1], minPath=1, maxPath=radius)
                    except:
                        ok = False
                        break
                if rid not in rfp:
                    rfp[rid] = fps[c[0]]
                else:
                    rfp[rid] = rfp[rid] ^ fps[c[0]]
            if not ok:
                del rfp[rid]
                continue
    return rfp
