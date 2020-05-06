import faiss
import numpy as np

d = 128                         # dimensionality of the input vectors
nlist = 128                     # number of the buckets of the inverted index.
# parameters for product quantization
m = 64                          # number of subquantizers
nbits_per_idx = 8               # number of bits per quantization index
dsub = d/m                      # dimensionality of each subvector

nb = 100000                      # database size
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.

quantizer = faiss.IndexFlatL2(d)
quantizer.verbose = True
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits_per_idx)
index.verbose = True
index.train(xb)
coarse_centroids = [quantizer.xb.at(i) for i in range(quantizer.xb.size())]  # nlist * d = 128 * 128 = 2^14 = 16K
pq_centroids = [index.pq.centroids.at(i) for i in range(index.pq.centroids.size())]  # m * 2^nbits_per_idx * dsub  = 2^15 = 32K
print((coarse_centroids, pq_centroids, index.pq.ksub, index.pq.dsub))

# Populate the index
index.add(xb)

# Perform a search
nq = 10     # nb of queries
k = 3
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.
D, I = index.search (xq, k)
print(D, I)
