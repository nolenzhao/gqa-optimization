
# GQA Kernel Optimization Scratchpad


## TODO: 
- Look into instructions to avoid padding on host and ensure out-of-bounds indexing returns zeros  
- Explore ping-ponging
- Look into changing B parameter of mfma instruction
- Caching vs non-caching loads
- Launch multiple blocks in M direction of A matrix
- Check for bank conflicts
- Split K loop iteration
- Increase occupancy to hide latency
- Implement half_lds usae with 4x4 -> do VGPR's/thread calculation to validate