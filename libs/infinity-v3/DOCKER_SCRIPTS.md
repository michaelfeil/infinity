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

CUDA inference in both Candle and LibTorch requires SM80 (Ampere) or newer.
SM75/Turing images are no longer built or supported. CPU builds are unaffected.

## Supported Variants

| Prefix | Architecture | Compute Cap | Dockerfile |
|--------|--------------|-------------|------------|
| (none) | Ampere | sm80 | Dockerfile-cuda |
| `86-` | A10 | sm86 | Dockerfile-cuda |
| `89-` | RTX 4000 | sm89 | Dockerfile-cuda |
| `hopper-` | Hopper | sm90 | Dockerfile-cuda |
| `blackwell-` | Datacenter Blackwell | sm100 | Dockerfile-cuda |
| `sm120-` | RTX Blackwell | sm120 | Dockerfile-cuda |

## Image Naming Convention

Images are tagged following the GitHub Actions conventions:

**Standard Images:**
- `{prefix}latest` - Latest version
- `{prefix}1.8.4` - Full version
- `{prefix}1.8` - Major.minor version

**Examples:**
- `baseten/infinity_v3:latest` (sm80)
- `baseten/infinity_v3:blackwell-latest` (sm100)
- `baseten/infinity_v3:sm120-latest` (sm120)

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
- `CUDA_COMPUTE_CAP` - Target compute capability
- `GIT_SHA` - Git commit SHA
- `SCCACHE_GHA_ENABLED` - Enable sccache caching

**Platform:**
- `linux/amd64` - All images built for this platform
- Multi-platform support can be added if needed

## Troubleshooting

### Common Issues

1. **Permission Denied:**
   ```bash
   chmod +x push-docker-images.sh quick-docker-build.sh
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

- **Current Version:** 1.8.4
- **Git SHA:** `git rev-parse --short HEAD`
- **Matrix Config:** `.github/workflows/matrix.json`

## Integration with CI/CD

These scripts follow the same patterns as the GitHub Actions workflow in `.github/workflows/build.yaml`. They can be used for:

- Local testing before CI runs
- Manual image builds for specific variants
- Emergency deployments
- Development and debugging

## Performance Tips

1. **Use sccache:** Enabled by default for faster compilation
2. **Parallel builds:** Use `--parallel` flag for multiple variants
3. **Build caching:** Inline cache enabled by default
4. **Selective builds:** Build only needed variants with quick script

## Security Notes

- Images are built with security scanning in CI
- No secrets are embedded in images
- Minimal base images used
- Regular security updates applied
