// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 IBM. Adapted by Infinity Core Contributors; see provenance.json.
// Extracted non-GDS reader; Python, GDS, dynamic loading and NUMA removed.
#include "nogds_reader.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
namespace fastsafetensors {
static constexpr int64_t ALIGN = 4096;
struct nogds_file_reader::state {
    struct request {
        int fd;
        uintptr_t destination;
        int64_t offset, length, next = 0;
        uint64_t remaining, prefix = 0;
        std::vector<uint8_t> completed;
        bool failed = false;
    };
    const ext_funcs_t *fns;
    int device;
    bool use_mmap, streams;
    static constexpr size_t buffers_per_thread = 2;
    uint64_t block_size, buffer_stride, allocation_size = 0;
    void *allocation = nullptr;
    std::vector<cudaStream_t> copy_streams;
    std::vector<std::thread> workers;
    std::mutex mutex;
#ifdef _MSC_VER
    std::mutex file_mutex;
#endif
    std::condition_variable ready, done;
    std::deque<std::shared_ptr<request>> queue;
    std::map<int, std::shared_ptr<request>> requests;
    bool stopping = false;
    int next_id = 1;

    static void check(cudaError_t error) {
        if (error != cudaSuccess)
            throw std::runtime_error("fastsafetensors operation failed: " + std::to_string(error));
    }
    void read(const request &req, int64_t position, int64_t length,
              char *bounce, cudaStream_t stream) {
        const int64_t offset = req.offset + position;
        int64_t source_skip = 0;
        if (use_mmap) {
#ifdef _MSC_VER
            constexpr int granularity = 65536;
#else
            constexpr int granularity = ALIGN;
#endif
            const int64_t map_offset = offset - offset % granularity;
            const size_t map_length = length + offset - map_offset;
            void *source = mmap(nullptr, map_length, PROT_READ, MAP_PRIVATE, req.fd, map_offset);
            if (source == MAP_FAILED) throw std::runtime_error("nogds mmap failed");
            std::memcpy(bounce, static_cast<char *>(source) + offset - map_offset, length);
            munmap(source, map_length);
        } else {
            int64_t read_offset = offset, read_length = length;
            bool direct = false;
#ifndef _MSC_VER
            const int flags = fcntl(req.fd, F_GETFL);
            direct = flags >= 0 && (flags & O_DIRECT) != 0;
            if (direct) {
                source_skip = offset % ALIGN;
                read_offset -= source_skip;
                read_length = ((length + source_skip + ALIGN - 1) / ALIGN) * ALIGN;
            }
#endif
            int64_t got;
            do {
#ifdef _MSC_VER
                std::lock_guard<std::mutex> file_lock(file_mutex);
#endif
                got = pread(req.fd, bounce, read_length, read_offset);
            } while (got < 0 && errno == EINTR);
            int read_errno = got < 0 ? errno : 0;
#ifndef _MSC_VER
            if (got < 0 && (read_errno == EINVAL || read_errno == EOPNOTSUPP || read_errno == EPERM) && direct
                    && fcntl(req.fd, F_SETFL, flags & ~O_DIRECT) == 0) {
                // The filesystem accepted O_DIRECT at open but refuses direct
                // reads. Clear the flag on the shared descriptor (every worker
                // sees the change) and retry buffered with
                // the exact range, as a buffered load would have issued.
                static std::atomic<bool> warned{false};
                if (!warned.exchange(true))
                    std::fprintf(stderr, "[WARN] filesystem rejected O_DIRECT reads: falling back to buffered reads\n");
                direct = false;
                source_skip = 0;
                do {
                    got = pread(req.fd, bounce, length, offset);
                } while (got < 0 && errno == EINTR);
                read_errno = got < 0 ? errno : 0;
            }
#endif
            // The aligned final read may extend past EOF; all requested tensor
            // bytes must nevertheless be present.
            if (got < length + source_skip) {
                std::string what = "nogds read failed or truncated input";
                if (got < 0) what += std::string(": ") + std::strerror(read_errno);
                if (direct) what += " (reads used O_DIRECT; disable direct I/O to use buffered reads)";
                throw std::runtime_error(what);
            }
        }
        void *destination = reinterpret_cast<void *>(req.destination + position);
        if (streams) {
            check(fns->cudaMemcpyAsync(destination, bounce + source_skip, length, cudaMemcpyHostToDevice, stream));
        } else {
            check(fns->cudaMemcpy(destination, bounce + source_skip, length, cudaMemcpyHostToDevice));
            check(fns->cudaDeviceSynchronize());
        }
    }
    void complete(const std::shared_ptr<request> &req, uint64_t position, bool failed) {
        std::lock_guard<std::mutex> lock(mutex);
        req->failed |= failed;
        const uint64_t old_prefix = req->prefix;
        if (!req->completed.empty() && !failed) {
            req->completed[position / block_size] = 1;
            // Workers can finish out of order. Publish only a contiguous prefix
            // whose DMA has completed, never merely scheduled bytes.
            while (req->prefix < req->completed.size() && req->completed[req->prefix])
                ++req->prefix;
        }
        --req->remaining;
        if (req->remaining == 0 || req->failed || old_prefix != req->prefix)
            done.notify_all();
    }
    void worker(size_t index) {
        const cudaError_t device_error = fns->cudaSetDevice(device);
        std::vector<std::shared_ptr<request>> pending(buffers_per_thread);
        std::vector<uint64_t> pending_positions(buffers_per_thread);
        const size_t first_slot = index * buffers_per_thread;
        size_t slot = 0;
        auto drain = [&](size_t i) {
            if (!pending[i]) return;
            bool failed = fns->cudaStreamSynchronize(copy_streams[first_slot + i]) != cudaSuccess;
            complete(pending[i], pending_positions[i], failed);
            pending[i].reset();
        };
        for (;;) {
            std::shared_ptr<request> req;
            int64_t position, length;
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (queue.empty()) {
                    // Finish pending DMA before sleeping, so wait_read can finish.
                    lock.unlock();
                    for (size_t i = 0; i < buffers_per_thread; ++i) drain(i);
                    lock.lock();
                }
                ready.wait(lock, [&] { return stopping || !queue.empty(); });
                if (queue.empty()) break;
                req = queue.front();
                position = req->next;
                length = std::min<int64_t>(block_size, req->length - position);
                req->next += length;
                if (req->next == req->length) queue.pop_front();
            }
            drain(slot);
            bool failed = false;
            try {
                check(device_error);
                const size_t i = first_slot + slot;
                read(*req, position, length, static_cast<char *>(allocation) + buffer_stride * i, copy_streams[i]);
            } catch (const std::exception &error) {
                // Drain any queued DMA before either source or destination reuse.
                if (streams) fns->cudaStreamSynchronize(copy_streams[first_slot + slot]);
                else fns->cudaDeviceSynchronize();
                std::fprintf(stderr, "%s\n", error.what());
                failed = true;
            }
            if (streams && !failed) {
                pending[slot] = req;
                pending_positions[slot] = position;
            } else complete(req, position, failed);
            slot = (slot + 1) % buffers_per_thread;
        }
    }
    ~state() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        ready.notify_all();
        for (auto &worker : workers) if (worker.joinable()) worker.join();
        fns->cudaSetDevice(device);
        if (streams) for (auto stream : copy_streams) fns->cudaStreamDestroy(stream);
        if (allocation) {
            fns->cudaFreeHost(allocation);
        }
    }
};

nogds_file_reader::nogds_file_reader(bool use_mmap, uint64_t bbuf_size_kb,
        uint64_t max_threads, const ext_funcs_t &functions, int device_id)
        : _state(new state) {
    state &s = *_state;
    s.fns = &functions;
    s.device = device_id;
    s.streams = s.fns->cudaMemcpyAsync && s.fns->cudaStreamCreateWithFlags
        && s.fns->cudaStreamSynchronize && s.fns->cudaStreamDestroy;
    if (bbuf_size_kb == 0 || max_threads == 0)
        throw std::invalid_argument("bounce buffer size and thread count must be positive");
    if (bbuf_size_kb > (1ULL << 30) || max_threads > 1024)
        throw std::invalid_argument("nogds buffer size or thread count is too large");
    const uint64_t slots = max_threads * state::buffers_per_thread;
    s.block_size = ((bbuf_size_kb + slots - 1) / slots) * 1024;
    s.use_mmap = use_mmap;
    state::check(s.fns->cudaSetDevice(device_id));
    s.buffer_stride = ((s.block_size + ALIGN - 1) / ALIGN) * ALIGN + 2 * ALIGN;
    s.allocation_size = s.buffer_stride * slots;
    cudaError_t allocation_error = s.fns->cudaSetDevice(device_id);
    if (allocation_error == cudaSuccess)
        allocation_error = s.fns->cudaHostAlloc(&s.allocation, s.allocation_size, 0);
    state::check(allocation_error);
    for (uint64_t i = 0; i < slots; ++i) {
        cudaStream_t stream = nullptr;
        if (s.streams) state::check(s.fns->cudaStreamCreateWithFlags(&stream, 1));
        s.copy_streams.push_back(stream);
    }
    for (uint64_t i = 0; i < max_threads; ++i)
        s.workers.emplace_back([&s, i] { s.worker(i); });
}

int nogds_file_reader::submit_read(int fd, const gds_device_buffer &dst,
        int64_t offset, int64_t length, uint64_t ptr_off, bool track_progress) {
    if (offset < 0 || length < 0 || offset > std::numeric_limits<int64_t>::max() - length)
        throw std::invalid_argument("invalid read offset or length");
    if (ptr_off > dst.get_length() || static_cast<uint64_t>(length) > dst.get_length() - ptr_off)
        throw std::out_of_range("destination out of bounds");
    state &s = *_state;
    if (s.use_mmap) {
#ifdef _MSC_VER
        struct _stat64 info{};
        if (_fstat64(fd, &info) || offset + length > info.st_size)
#else
        struct stat info{};
        if (fstat(fd, &info) || offset + length > info.st_size)
#endif
            throw std::runtime_error("nogds mmap input is truncated");
    }
    state::check(s.fns->cudaSetDevice(s.device));
    // Fence after framework allocation: cached storage can have a pending clone.
    state::check(s.fns->cudaDeviceSynchronize());
    auto req = std::make_shared<state::request>();
    req->fd = fd; req->destination = dst.get_base_address() + ptr_off;
    req->offset = offset; req->length = length;
    req->remaining = length ? (length - 1) / s.block_size + 1 : 0;
    if (track_progress) req->completed.resize(req->remaining, 0);
    int id;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.next_id == std::numeric_limits<int>::max()) throw std::overflow_error("request id overflow");
        id = s.next_id++;
        s.requests[id] = req;
        if (length) s.queue.push_back(req);
    }
    s.ready.notify_all();
    return id;
}

uint64_t nogds_file_reader::wait_read_prefix(int id, uint64_t length) {
    state &s = *_state;
    std::unique_lock<std::mutex> lock(s.mutex);
    auto req = s.requests.at(id);
    if (length > static_cast<uint64_t>(req->length))
        throw std::out_of_range("read prefix out of bounds");
    if (req->length && req->completed.empty())
        throw std::invalid_argument("read progress was not enabled");
    const uint64_t blocks = length ? (length - 1) / s.block_size + 1 : 0;
    s.done.wait(lock, [&] { return req->failed || req->prefix >= blocks; });
    if (req->failed) throw std::runtime_error("nogds read failed or truncated input");
    return std::min<uint64_t>(req->length, req->prefix * s.block_size);
}

uintptr_t nogds_file_reader::wait_read(int id) {
    state &s = *_state;
    std::unique_lock<std::mutex> lock(s.mutex);
    auto req = s.requests.at(id);
    s.done.wait(lock, [&] { return req->remaining == 0; });
    s.requests.erase(id);
    return req->failed ? 0 : req->destination;
}

nogds_file_reader::~nogds_file_reader() = default;
} // namespace fastsafetensors
