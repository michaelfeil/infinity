# Docker Build Script

This directory contains a comprehensive script for building and pushing Docker images for infinity_v3 with multiple CUDA compute capabilities.

## Script Overview

### `push-docker-images.sh` (Comprehensive)
Full-featured script with error handling, parallel builds, and verification.

**Features:**
- Configurable image repository (`IMAGE_REPOSITORY`)
- Parallel builds for faster execution
- Image verification after pushing
- Comprehensive logging and error handling
- Dry-run mode for testing
- Support for all variants including Blackwell (sm120)

**Usage:**
```bash
# Build all variants sequentially
./push-docker-images.sh

# Build in parallel (faster)
./push-docker-images.sh --parallel

# Dry run to see what would be built
./push-docker-images.sh --dry-run

# Verify existing images only
./push-docker-images.sh --verify

# Show help
./push-docker-images.sh --help
```

CUDA inference in LibTorch requires SM80 (Ampere) or newer.
SM75/Turing images are no longer built or supported. CPU builds are unaffected.

## Supported Variants

| Prefix | Architecture | Compute Cap | Dockerfile |
|--------|--------------|-------------|------------|
| `nvidia-` | Ampere through Blackwell | sm80, sm86, sm89, sm90, sm100, sm120 | Dockerfile-libtorch |
| `cpu-` | CPU | n/a | Dockerfile |

## Image Naming Convention

Images are tagged following the GitHub Actions conventions:

**Standard Images:**
- `{prefix}latest` - Latest version
- `{prefix}3.0.1` - Full version
- `{prefix}3.0` - Major.minor version

**Examples:**
- `baseten/infinity_v3:nvidia-latest` (all supported GPUs)
- `baseten/infinity_v3:cpu-latest` (CPU)

## Registries

The script defaults to `baseten/infinity_v3`. Override the destination with
`IMAGE_REPOSITORY`, for example:

```bash
IMAGE_REPOSITORY=ghcr.io/my-org/infinity_v3 ./push-docker-images.sh --dry-run
```

GitHub Actions uses `ghcr.io/${{ github.repository_owner }}/infinity_v3`.
These are configured destinations; publishing requires registry credentials.

## Prerequisites

1. **Docker & Docker Buildx:**
   ```bash
   docker --version
   docker buildx version
   ```

2. **Registry Access:**
   ```bash
   docker login ghcr.io
   # For the default Docker Hub destination:
   docker login # Docker Hub, for the default baseten/infinity_v3 destination
   ```

3. **Git Repository:**
   - Scripts must be run from the repository root
   - Clean git state recommended

## Environment Variables

Optional environment variables for better performance:

```bash
export DOCKER_BUILDKIT=1
export BUILDKIT_INLINE_CACHE=1
```

## Build Configuration

**Common Build Args:**
- `LIBTORCH_CUDA_ARCHITECTURES` - NVIDIA targets (defaults to all supported architectures)
- `GIT_SHA` - Git commit SHA
- `BUILD_JOBS` - Compiler parallelism

**Platform:**
- `linux/amd64` - All images built for this platform
- Multi-platform support can be added if needed

## Troubleshooting

### Common Issues

1. **Permission Denied:**
   ```bash
   chmod +x push-docker-images.sh
   ```

2. **Registry Login:**
   ```bash
   docker logout ghcr.io
   docker login ghcr.io
   ```

3. **Buildx Not Available:**
   ```bash
   docker buildx install
   docker buildx create --use
   ```

4. **Out of Memory:**
   - Reduce parallel builds
   - Use sequential mode: `./push-docker-images.sh`

### Debug Mode

Enable verbose logging:
```bash
export DOCKER_BUILDKIT=1
docker buildx build --progress=plain ...
```

## Version Information

- **Current Version:** 3.0.1
- **Git SHA:** `git rev-parse --short HEAD`
- **Matrix Config:** `.github/infinity-v3-matrix.json`

## Integration with CI/CD

These scripts follow the same patterns as the GitHub Actions workflow in `.github/workflows/infinity-v3-docker.yaml`. They can be used for:

- Local testing before CI runs
- Manual image builds for specific variants
- Emergency deployments
- Development and debugging

## Performance Tips

2. **Parallel builds:** Use `--parallel` flag for multiple variants
3. **Build caching:** Inline cache enabled by default
4. **Selective builds:** Build the CPU or NVIDIA Dockerfile directly

## Security Notes

- Images are built with security scanning in CI
- No secrets are embedded in images
- Minimal base images used
- Regular security updates applied
