import random

def IsValid(seq, match = {'(':')', '[':']', '{':'}', '<':'>'}, 
            pad = '.') -> bool:
    """
    Given an input sequence, return whether it is correct.
    """
    stack = []
    for idx, char in enumerate(seq):
        if char in match.keys():
            stack.append(char)
        elif char in match.values():
            if len(stack) == 0:
                return False
            if char == match[stack[-1]]:
                stack.pop()
                continue
            else:
                return False
        else:
            return (idx > 0 and 
                    all(i == pad for i in seq[idx:]) and 
                    len(stack) == 0)
    return len(stack) == 0

def SampleCorrect(n, match = {'(':')', '[':']', '{':'}', '<':'>'}, 
                  pad = '.', MAX_LEN = 10) -> set:
    """
    Create a set of gramatically correct sequences. Works faster than purely random generation.
    """
    res = set()
    while len(res) < n:
        stack = []
        pool = list(match.keys()) + [pad]
        newstr = ""
        while len(newstr) < MAX_LEN:
            if len(stack) == 0:
                toadd = random.choices(pool, k=1)[0]
            else:
                toadd = random.choices(pool + [match[stack[-1]]], k=1)[0]
            if toadd in match.keys():
                stack.append(toadd)
            elif toadd in match.values():
                stack.pop()
            elif toadd == pad:
                newstr += pad*(MAX_LEN - len(newstr))
                break
            newstr += toadd
        if IsValid(newstr):
            res.add(newstr)
    return res

def RealLen(seqs, pad = '.', 
            MAX_LEN = 10) -> list:
    """
    For dataset investigation purposes. 
    Gives the real lenghts (ie without padding) of a list of sequences.
    """
    lens = []
    for seq in seqs:
        for idx, char in enumerate(seq):
            if char == pad:
                lens.append(idx)
                break
            if idx == MAX_LEN - 1:
                lens.append(MAX_LEN)
    return lens

def count_parameters(model) -> int: 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def performance(outs, groundt, treshold = 0.5) -> dict:
    truepos, trueneg, falseneg, falsepos = 0, 0, 0, 0
    preds = [int(out.item() > treshold) for out in outs.view(-1)]
    for i in range(len(preds)):
        if preds[i] == int(groundt[i].item()):
            if preds[i] == 1:
                truepos += 1
            else:
                trueneg += 1
        elif int(groundt[i].item()) == 1 and preds[i] ==0:
            falseneg += 1
        else:
            falsepos += 1
    precision = truepos/(truepos + falsepos)
    recall = truepos/(truepos + falseneg)
    accuracy = (truepos+trueneg)/len(preds)
    Fscore = 2*precision*recall/(precision+recall)
    perf = {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "Fscore": Fscore,
            }
    return perf
