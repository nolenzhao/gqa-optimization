
# GQA Kernel Optimization Scratchpad


## TODO: 
- Reduce mfma size -> use 4x4 to fully occupy matrix when GROUP_SIZE is 4, or 8
- Look into instructions to avoid padding on host and ensure out-of-bounds indexing returns zeros  
- Explore ping-ponging