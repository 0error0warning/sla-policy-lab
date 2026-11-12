"""Lossless storage for the large [partner, health, language key, hidden] matrices.

Two actual rows predict the other three. Their exact XOR residuals are stored,
including all rounding bits. Decoding uses bounded integer dot products and powers
of two, never model replay or a floating-point least-squares fit.
"""
from hashlib import sha256
from itertools import combinations
import json
import numpy as np

PAIRS=tuple(combinations(range(5),2))


def byte_planes(a):
    a=np.ascontiguousarray(a)
    return a.reshape(-1).view(np.uint8).reshape(-1,a.dtype.itemsize).T.copy()


def unplanes(data,dtype,shape):
    return np.ascontiguousarray(data.T).reshape(-1).view(dtype).reshape(shape)


def exponent(x,bits):
    return int(np.ceil(np.log2(max(float(np.max(np.abs(x))),1e-30))))-bits


def predict(anchor,cq,e1,e2):
    xi=np.rint(np.ldexp(anchor.astype(np.float64),-int(e1))).astype(np.int64)
    prod=cq.astype(np.int64)@xi
    assert np.max(np.abs(prod))<2**53
    return np.ldexp(prod.astype(np.float64),int(e1)+int(e2)).astype("<f4")


def pack(source,destination):
    entries=[];extras={};groups=[]
    with np.load(source,allow_pickle=False) as z:
        for i,name in enumerate(z.files):
            a=np.ascontiguousarray(z[name])
            entry=dict(name=name,shape=list(a.shape),dtype=a.dtype.str,sha256=sha256(a.tobytes()).hexdigest())
            if a.dtype==np.dtype("<f4") and a.ndim==4 and a.shape[2]==5:
                entry["block_count"]=a.shape[0]*a.shape[1]
                groups.extend(a.reshape(-1,5,a.shape[-1]))
            else:
                entry["extra"]=f"x{i}";extras[entry["extra"]]=a
            entries.append(entry)
    if not groups:raise ValueError("No personal matrices to pack")
    anchor=[];residue=[];coeff=[];orders=[];scale=[]
    for a in groups:
        gram=a.astype(np.float64)@a.astype(np.float64).T
        pair=max(PAIRS,key=lambda ij:gram[ij[0],ij[0]]*gram[ij[1],ij[1]]-gram[ij]**2)
        rest=[i for i in range(5) if i not in pair]
        w=a[list(pair)]
        c=gram[np.ix_(rest,pair)]@np.linalg.pinv(gram[np.ix_(pair,pair)])
        e1,e2=exponent(w,25),exponent(c,23)
        cq=np.rint(np.ldexp(c,-e2)).astype("<i4")
        pred=predict(w,cq,e1,e2)
        anchor.append(w);residue.append(a[rest].view("<u4")^pred.view("<u4"));coeff.append(cq)
        orders.append(pair);scale.append((e1,e2))
    width=groups[0].shape[-1]
    np.savez_compressed(destination,metadata=np.array(json.dumps(dict(entries=entries,width=width,version=1))),
        anchor=byte_planes(np.stack(anchor)),residue=byte_planes(np.stack(residue)),
        coeff=np.asarray(coeff),order=np.asarray(orders,np.uint8),scale=np.asarray(scale,np.int16),**extras)


def restore_arrays(compact):
    with np.load(compact,allow_pickle=False) as z:
        meta=json.loads(str(z["metadata"]));assert meta["version"]==1
        order=z["order"];width=meta["width"];count=len(order)
        anchors=unplanes(z["anchor"],"<f4",(count,2,width))
        errors=unplanes(z["residue"],"<u4",(count,3,width))
        blocks=np.empty((count,5,width),"<f4")
        for i,(a,delta,cq,pair,powers) in enumerate(zip(anchors,errors,z["coeff"],order,z["scale"])):
            rest=[j for j in range(5) if j not in pair]
            blocks[i,pair]=a
            blocks[i,rest]=(delta^predict(a,cq,*powers).view("<u4")).view("<f4")
        out={};offset=0
        for entry in meta["entries"]:
            if "extra" in entry:
                a=z[entry["extra"]]
            else:
                n=entry["block_count"];a=blocks[offset:offset+n].reshape(entry["shape"]);offset+=n
            assert a.dtype.str==entry["dtype"] and sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==entry["sha256"]
            out[entry["name"]]=a
        assert offset==count
        return out


def restore(compact,destination):
    np.savez_compressed(destination,**restore_arrays(compact))
