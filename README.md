
# GQA Kernel Optimization 

This repo contains different implementations of the Query-Key Transpose GEMM in the GQA self-attention mechanism in autoregressive mode. 
The kernels aim to optimize the GEMM in GQA by placing multiple query vectors in the first matrix, as in GQA multiple query vectors 
share a common key matrix (K cache). Thus isntead of performing `GROUP_SIZE` GEMM's -- only one is performed per group. 


## About Src
Each different implementation has an associated namespace which may implement different variations of loads/stores from/to HBM/LDS as well as 
different sizes of the MFMA instructions or workgroup occupancy. 

## Build
`hipcc -o /path/to/output/main -I/path/to/include/ /path/to/src/helpers.cpp /path/to/src/kernels.cpp /path/to/src/main.cpp -w`


## Run 
`/path/to/output/main`

## Future Work
- Look into instructions to avoid padding on host and ensure out-of-bounds indexing returns zeros  
- Check for bank conflicts and implement LDS padding if needed
- Move `atomicAdd` out the loop in the splitK implementation. Possibly use a sweeping reduction kernel instead of summing within GQA kernel
- Increase occupancy to hide latency
