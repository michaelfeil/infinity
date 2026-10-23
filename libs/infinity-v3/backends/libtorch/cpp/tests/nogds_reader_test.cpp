#include "vendor/fastsafetensors/nogds_reader.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <vector>

namespace {
void require(bool ok) { if (!ok) throw std::runtime_error("reader invariant failed"); }
struct Stream { void* dst = nullptr; const void* src = nullptr; size_t count = 0; };
int device(int) { return 0; }
int allocate(void** p, size_t n, unsigned) { return posix_memalign(p, 4096, n); }
int release(void* p) { free(p); return 0; }
int copy(void* dst, const void* src, size_t n, int) { memcpy(dst, src, n); return 0; }
int synchronize() { return 0; }
int create(void** p, unsigned) { *p = new Stream; return 0; }
int enqueue(void* dst, const void* src, size_t n, int, void* raw) {
    auto& stream = *static_cast<Stream*>(raw);
    require(!stream.count);
    stream = {dst, src, n}; // Deliberately borrow the bounce buffer until wait.
    return 0;
}
int wait(void* raw) {
    auto& stream = *static_cast<Stream*>(raw);
    if (stream.count) memcpy(stream.dst, stream.src, stream.count);
    stream.count = 0;
    return 0;
}
int destroy(void* raw) { wait(raw); delete static_cast<Stream*>(raw); return 0; }
const fastsafetensors::ext_funcs_t ops{device, allocate, release, copy, synchronize,
                                      enqueue, create, wait, destroy};
struct File {
    int fd;
    File() {
        char path[] = "/tmp/infinity-nogds-test-XXXXXX";
        fd = mkstemp(path);
        require(fd >= 0);
        unlink(path);
    }
    ~File() { close(fd); }
};
}
int main() {
    try {
        File source;
        std::vector<uint8_t> bytes((8 << 20) + 123);
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (i * 29 + i / 4096) % 251;
        require(write(source.fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()));
        // More blocks than buffer slots, and deferred copies that can observe
        // premature reuse. Multiple reads exercise out-of-order completion.
        std::vector<uint8_t> first(bytes.size() + 64, 0xa5), second(first.size(), 0xa5);
        {
            fastsafetensors::nogds_file_reader reader(false, 64, 4, ops, 0);
            fastsafetensors::gds_device_buffer a(reinterpret_cast<uintptr_t>(first.data()), bytes.size());
            fastsafetensors::gds_device_buffer b(reinterpret_cast<uintptr_t>(second.data()), bytes.size());
            int x = reader.submit_read(source.fd, a, 0, bytes.size(), 0, true);
            int y = reader.submit_read(source.fd, b, 0, bytes.size(), 0);
            require(reader.wait_read_prefix(x, 1) >= 1);
            require(reader.wait_read(x) == a.get_base_address());
            require(reader.wait_read(y) == b.get_base_address());
            bool rejected = false;
            try { reader.submit_read(source.fd, a, -1, 1, 0); } catch (const std::exception&) { rejected = true; }
            require(rejected);
            rejected = false;
            try { reader.submit_read(source.fd, a, 0, 2, bytes.size() - 1); } catch (const std::exception&) { rejected = true; }
            require(rejected);
            // EOF must fail after draining all other workers, never publish
            // partially initialized output as a successfully loaded checkpoint.
            require(ftruncate(source.fd, bytes.size() - 1) == 0);
            int truncated = reader.submit_read(source.fd, a, 0, bytes.size(), 0);
            require(reader.wait_read(truncated) == 0);
        }
        require(std::equal(bytes.begin(), bytes.end(), second.begin()));
        require(std::all_of(first.end()-64, first.end(), [](auto v) { return v == 0xa5; }));
        require(std::all_of(second.end()-64, second.end(), [](auto v) { return v == 0xa5; }));
        std::cout << "non-GDS worker lifetime, bounds and truncation checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
