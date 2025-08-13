
# GQA Kernel Optimization Scratchpad


## TODO: 
- Look into instructions to avoid padding on host and ensure out-of-bounds indexing returns zeros  
- Look into changing B parameter of mfma instruction
- Caching vs non-caching loads
- Check for bank conflicts
- Split K loop iteration
- Increase occupancy to hide latency