# Kernel selection

Prefer common, proven kernels such as FA4 across supported GPU families and providers.
Use a shared dispatch policy; avoid architecture-specific tuning and complicated fallback chains.
Add special paths only when required for correctness or an unsupported capability.
