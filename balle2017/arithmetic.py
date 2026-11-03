"""Reference binary arithmetic coder. NOT the H.264 CABAC engine.

Implements the paper's mode/sign/unary/Exp-Golomb decision tree with adaptive
contexts shared spatially per channel. The arithmetic engine and adaptation
are deliberately identified as a reproduction deviation.
"""
import numpy as np


class Bits:
    def __init__(self, data=None):
        self.bits = [] if data is None else [(b >> i) & 1 for b in data for i in range(7, -1, -1)]
        self.position = 0

    def read(self):
        bit = self.bits[self.position] if self.position < len(self.bits) else 0
        self.position += 1
        return bit

    def pack(self):
        padded = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(sum(padded[j+i] << (7-i) for i in range(8)) for j in range(0, len(padded), 8))


class Arithmetic:
    FULL, HALF, QUARTER = 1 << 32, 1 << 31, 1 << 30

    def __init__(self, data=None):
        self.decode = data is not None
        self.stream = Bits(data)
        self.low, self.high, self.pending = 0, self.FULL - 1, 0
        self.code = 0
        if self.decode:
            for _ in range(32):
                self.code = self.code * 2 + self.stream.read()

    def emit(self, bit):
        self.stream.bits.append(bit)
        self.stream.bits.extend([1-bit] * self.pending)
        self.pending = 0

    def binary(self, counts, value=None, adaptive=True):
        split = self.low + (self.high-self.low+1) * counts[0] // sum(counts)
        if self.decode:
            value = int(self.code >= split)
        if value:
            self.low = split
        else:
            self.high = split - 1
        while True:
            if self.high < self.HALF:
                if not self.decode:
                    self.emit(0)
            elif self.low >= self.HALF:
                if not self.decode:
                    self.emit(1)
                self.low -= self.HALF
                self.high -= self.HALF
                if self.decode:
                    self.code -= self.HALF
            elif self.low >= self.QUARTER and self.high < 3*self.QUARTER:
                self.pending += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
                if self.decode:
                    self.code -= self.QUARTER
            else:
                break
            self.low *= 2
            self.high = self.high * 2 + 1
            if self.decode:
                self.code = self.code * 2 + self.stream.read()
        if adaptive:
            counts[value] += 1
            if sum(counts) >= 32768:
                counts[:] = [max(1, (v+1)//2) for v in counts]
        return value

    def finish(self):
        self.pending += 1
        self.emit(int(self.low >= self.QUARTER))
        return self.stream.pack()

    def golomb(self, value=None):
        # Order-zero unsigned Exp-Golomb, bypass probability 1/2.
        if not self.decode:
            digits = bin(value + 1)[2:]
            for bit in [0] * (len(digits)-1) + [int(x) for x in digits]:
                self.binary([1, 1], bit, adaptive=False)
            return value
        zeros = 0
        while not self.binary([1, 1], adaptive=False):
            zeros += 1
            if zeros > 31:
                raise ValueError('Invalid Exp-Golomb value.')
        number = 1
        for _ in range(zeros):
            number = number * 2 + self.binary([1, 1], adaptive=False)
        return number - 1


def initial_contexts(probabilities):
    p = np.array(probabilities, dtype=np.float64, copy=True)
    p /= p.sum()
    mode = int(p.argmax())
    contexts = {}

    def add(key, probability_one):
        n1 = min(1023, max(1, int(round(float(probability_one)*1024))))
        contexts[key] = [1024-n1, n1]

    add('mode', p[mode])
    remaining = max(1e-15, 1-p[mode])
    add('sign', p[mode+1:].sum() / remaining)
    for sign in (-1, 1):
        for distance in range(1, len(p)):
            index = mode + sign*distance
            if not 0 <= index < len(p):
                break
            tail = p[index:].sum() if sign == 1 else p[:index+1].sum()
            # True means beyond this candidate (Figure 9 positive branch).
            add((sign, distance), (tail-p[index]) / max(tail, 1e-15))
    return mode, contexts


def symbol(coder, contexts, mode, minimum, maximum, value=None):
    if coder.binary(contexts['mode'], None if value is None else int(value == mode)):
        return mode
    positive = coder.binary(contexts['sign'], None if value is None else int(value > mode))
    sign = 1 if positive else -1
    distance = 1
    while minimum <= mode + sign*distance <= maximum:
        candidate = mode + sign*distance
        beyond = None if value is None else int((value-candidate)*sign > 0)
        if not coder.binary(contexts[(sign, distance)], beyond):
            return candidate
        distance += 1
    bound = maximum if positive else minimum
    residual = coder.golomb(None if value is None else abs(value-bound)-1)
    return bound + sign*(residual+1)


def encode_latents(q, integers, probabilities):
    coder = Arithmetic()
    for plane, probs in zip(q, probabilities):
        index, contexts = initial_contexts(probs)
        mode = int(integers[index])
        for value in plane.flat:
            symbol(coder, contexts, mode, int(integers[0]), int(integers[-1]), int(value))
    return coder.finish()


def decode_latents(data, shape, integers, probabilities):
    if len(shape) != 3 or shape[0] != len(probabilities):
        raise ValueError('Invalid latent shape.')
    q = np.empty(shape, dtype=np.int32)
    coder = Arithmetic(data)
    for plane, probs in zip(q, probabilities):
        index, contexts = initial_contexts(probs)
        for position in np.ndindex(plane.shape):
            plane[position] = symbol(coder, contexts, int(integers[index]), int(integers[0]), int(integers[-1]))
    return q
