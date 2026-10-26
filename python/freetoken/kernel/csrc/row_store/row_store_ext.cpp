// Disk-backed row store: fixed-width rows read straight from checkpoint shard tensors
// through an extent table (row i of extent e at extent_base[e] + i * row_stride).
// One caller thread at a time, no locks on the store itself; a large fill splits over
// several io_uring rings, each driven by its own helper thread. Duplicate rows in one fill
// dedup into ONE batched read round; an optional RAM cache keeps rows read before (cache_rows); no per-sequence
// state.
//
//   RowStore  -- hash-agnostic: stage_rows(row_ids) + flush(); callers supply row ids.
//   PleStore  -- RowStore plus the Qwen3.8-Flash-Next PLE n-gram hash: stage(tokens).
//                Hash reference: tests/models/qwen4_exp/test_ple_disk.py.
//
// Platform seams: TableFile (O_DIRECT+pread; Win: NO_BUFFERING), BatchReader (io_uring,
// pread-pool fallback = the portable shape), stream memops (CUDA driver or HIP signal memory).

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__linux__) && __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#define PLE_HAS_IO_URING 1
#else
#define PLE_HAS_IO_URING 0
#endif

#include <torch/extension.h>

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#include <hip/hip_runtime.h>
#endif

namespace py = pybind11;

namespace {

constexpr int64_t kPage = 4096;
constexpr unsigned kBatchEntries = 64;
// fio on this class of disk: pread saturates at ~16 threads; more only adds latency
constexpr unsigned kReaderThreads = 16;
// FREETOKEN_ROW_STORE_RINGS (default 4): io_uring rings a large fill splits over, one thread
// each. One thread tops out at ~280k reads/s (its submit / complete loop, whatever the depth);
// the reference NVMe takes ~490k/s from four
constexpr unsigned kRingsDefault = 4;
constexpr unsigned kRingsMax = 16;
// below this many rows a fill stays on one ring: the helpers' wake-up would cost more than it saves
constexpr size_t kSplitMinRows = 256;

// ---- portability shims ----

void release_store_i64(int64_t *ptr, int64_t value) {
  __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

uint8_t *page_aligned_alloc(size_t bytes) {
  void *p = nullptr;
  if (posix_memalign(&p, kPage, bytes) != 0) throw std::bad_alloc();
  return static_cast<uint8_t *>(p);
}

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

hipError_t alloc_signal(int64_t **out) {
  void *word = nullptr;
  // Like CPU MoE, HIP stream waits require signal memory, not a pinned staging allocation.
  const auto rc = hipExtMallocWithFlags(&word, sizeof(int64_t), hipMallocSignalMemory);
  if (rc == hipSuccess) {
    *out = static_cast<int64_t *>(word);
    release_store_i64(*out, 0);
  }
  return rc;
}

torch::Tensor alloc_signal_tensor() {
  int64_t *word = nullptr;
  const auto rc = alloc_signal(&word);
  TORCH_CHECK(rc == hipSuccess, "HIP signal allocation failed: ", hipGetErrorString(rc));
  return torch::from_blob(word, {1}, [](void *p) { (void)hipFree(p); },
                          torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
}

int memop_write(uintptr_t stream, uintptr_t addr, uint64_t value) {
  return hipStreamWriteValue64(reinterpret_cast<hipStream_t>(stream),
                               reinterpret_cast<void *>(addr), value, hipStreamWriteValueDefault);
}

int memop_wait_geq(uintptr_t stream, uintptr_t addr, uint64_t value) {
  return hipStreamWaitValue64(reinterpret_cast<hipStream_t>(stream),
                              reinterpret_cast<void *>(addr), value, hipStreamWaitValueGte, UINT64_MAX);
}

// Capability failures must not poison gating fallback or hide a caller's earlier error.
struct ProbeErrorScope {
  ProbeErrorScope() {
    TORCH_CHECK(hipPeekAtLastError() == hipSuccess, "HIP error pending before row-store sync probe");
  }
  ~ProbeErrorScope() { (void)hipGetLastError(); }
};

bool capture_memops_supported() {
  ProbeErrorScope errors;
  int device = 0;
  int can_wait = 0;
  if (hipGetDevice(&device) != hipSuccess ||
      hipDeviceGetAttribute(&can_wait, hipDeviceAttributeCanUseStreamWaitValue, device) != hipSuccess || can_wait == 0)
    return false;

  int64_t *ready = nullptr;
  int64_t *done = nullptr;
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  auto rc = alloc_signal(&ready);
  if (rc == hipSuccess) rc = alloc_signal(&done);
  if (rc == hipSuccess) rc = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  // A runtime that executes the WAIT during capture must not block the probe.
  if (rc == hipSuccess) release_store_i64(done, 1);
  if (rc == hipSuccess) rc = hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed);
  const bool started = rc == hipSuccess;
  if (rc == hipSuccess) rc = static_cast<hipError_t>(memop_write(
      reinterpret_cast<uintptr_t>(stream), reinterpret_cast<uintptr_t>(ready), 1));
  if (rc == hipSuccess) rc = static_cast<hipError_t>(memop_wait_geq(
      reinterpret_cast<uintptr_t>(stream), reinterpret_cast<uintptr_t>(done), 1));
  if (started) {
    const auto end_rc = hipStreamEndCapture(stream, &graph);
    if (rc == hipSuccess) rc = end_rc;
  }
  if (rc == hipSuccess) rc = hipStreamSynchronize(stream);
  size_t nodes = 0;
  if (rc == hipSuccess) rc = hipGraphGetNodes(graph, nullptr, &nodes);
  // Older HIP runtimes execute memops during capture without recording them.
  if (rc == hipSuccess && nodes != 2) rc = hipErrorNotSupported;
  if (rc == hipSuccess) rc = hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  bool served = false;
  if (rc == hipSuccess) {
    release_store_i64(ready, 0);
    release_store_i64(done, 0);
    std::thread responder([&] {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (__atomic_load_n(ready, __ATOMIC_ACQUIRE) == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      served = __atomic_load_n(ready, __ATOMIC_ACQUIRE) == 1;
      // The graph never resets done, so even a launch arriving after the deadline is released.
      release_store_i64(done, 1);
    });
    rc = hipGraphLaunch(exec, stream);
    if (rc == hipSuccess) rc = hipStreamSynchronize(stream);
    responder.join();
  }
  const bool ok = rc == hipSuccess && served;
  if (exec != nullptr) (void)hipGraphExecDestroy(exec);
  if (graph != nullptr) (void)hipGraphDestroy(graph);
  if (stream != nullptr) {
    (void)hipStreamSynchronize(stream);
    (void)hipStreamDestroy(stream);
  }
  if (ready != nullptr) (void)hipFree(ready);
  if (done != nullptr) (void)hipFree(done);
  return ok;
}

#else

// Stream memops for the flag-sync fast path, resolved from the driver at runtime.
using CuMemOp64Fn = int (*)(void *stream, unsigned long long addr, unsigned long long value,
                            unsigned int flags);
CuMemOp64Fn g_cu_write64 = nullptr;
CuMemOp64Fn g_cu_wait64 = nullptr;
constexpr unsigned kCuWaitValueGeq = 0x0;
constexpr unsigned kCuWriteDefault = 0x0;

bool cumemop_resolve() {
  static bool resolved = [] {
    void *h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (h == nullptr) h = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
    if (h == nullptr) return false;
    g_cu_write64 = reinterpret_cast<CuMemOp64Fn>(dlsym(h, "cuStreamWriteValue64_v2"));
    if (g_cu_write64 == nullptr)
      g_cu_write64 = reinterpret_cast<CuMemOp64Fn>(dlsym(h, "cuStreamWriteValue64"));
    g_cu_wait64 = reinterpret_cast<CuMemOp64Fn>(dlsym(h, "cuStreamWaitValue64_v2"));
    if (g_cu_wait64 == nullptr)
      g_cu_wait64 = reinterpret_cast<CuMemOp64Fn>(dlsym(h, "cuStreamWaitValue64"));
    return g_cu_write64 != nullptr && g_cu_wait64 != nullptr;
  }();
  return resolved;
}

int memop_write(uintptr_t stream, uintptr_t addr, uint64_t value) {
  if (!cumemop_resolve()) return -1;
  return g_cu_write64(reinterpret_cast<void *>(stream), addr, value, kCuWriteDefault);
}

int memop_wait_geq(uintptr_t stream, uintptr_t addr, uint64_t value) {
  if (!cumemop_resolve()) return -1;
  return g_cu_wait64(reinterpret_cast<void *>(stream), addr, value, kCuWaitValueGeq);
}

#endif

// WAIT(>=1) then RESET: resetting first would race a fast host signal and deadlock the stream.
void memop_wait_reset(uintptr_t stream, uintptr_t flag_addr) {
  if (memop_wait_geq(stream, flag_addr, 1) != 0 || memop_write(stream, flag_addr, 0) != 0)
    throw std::runtime_error("stream memops rejected in capture; set FREETOKEN_PLE_SYNC=gate");
}

void signal_flag(uintptr_t flag_addr) {
  release_store_i64(reinterpret_cast<int64_t *>(flag_addr), 1);
}

// ---- TableFile: platform seam for on-disk files ----

// Read at least need bytes; len is the larger aligned span the request must keep.
// Resuming past need is not safe: a read that crossed EOF ends at an unaligned offset.
void pread_min(int fd, uint8_t *buf, int64_t len, int64_t need, int64_t off) {
  int64_t done = 0;
  while (done < need) {
    ssize_t got = ::pread(fd, buf + done, len - done, off + done);
    if (got < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("pread: ") + std::strerror(errno));
    }
    if (got == 0) break;
    done += got;
  }
  if (done < need)
    throw std::runtime_error("short read at offset " + std::to_string(off) + ": got " +
                             std::to_string(done) + " of " + std::to_string(need));
}

class TableFile {
 public:
  explicit TableFile(const std::string &path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    direct_ = fd_ >= 0;
    if (fd_ < 0) {
      fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
      direct_ = false;
    }
    if (fd_ < 0) throw std::runtime_error(path + ": " + std::strerror(errno));
    struct stat st{};
    if (fstat(fd_, &st) != 0) {
      ::close(fd_);
      throw std::runtime_error(path + ": fstat: " + std::strerror(errno));
    }
    size_ = st.st_size;
    if (!direct_) posix_fadvise(fd_, 0, 0, POSIX_FADV_RANDOM);
  }

  ~TableFile() {
    if (fd_ >= 0) ::close(fd_);
  }

  TableFile(const TableFile &) = delete;
  TableFile &operator=(const TableFile &) = delete;

  bool direct_io() const { return direct_; }
  int native_fd() const { return fd_; }
  int64_t size() const { return size_; }

  // keep buffered fallback reads out of the page cache; a no-op under direct I/O
  void discard_cache(int64_t off, int64_t len) const {
    if (!direct_) posix_fadvise(fd_, off, len, POSIX_FADV_DONTNEED);
  }

 private:
  int fd_ = -1;
  bool direct_ = false;
  int64_t size_ = 0;
};

// ---- BatchReader: platform seam for parallel positioned reads ----

// Pipelined: at most capacity() reads in flight; wait_one() returns a finished tag to refill.
class BatchReader {
 public:
  virtual ~BatchReader() = default;
  virtual std::string name() const = 0;
  virtual unsigned capacity() const = 0;
  virtual void submit(unsigned tag, int fd, uint8_t *buf, int64_t len, int64_t need,
                      int64_t off) = 0;
  virtual unsigned wait_one() = 0;
  // reap every in-flight read so stale completions cannot leak into the next fill
  virtual void drain() noexcept = 0;
};

class ThreadPoolBatchReader final : public BatchReader {
 public:
  ThreadPoolBatchReader() {
    // these threads block on I/O, not compute, so the core count is only a default
    unsigned n = std::max(1u, std::min(kReaderThreads, std::thread::hardware_concurrency()));
    const char *env = std::getenv("FREETOKEN_ROW_STORE_READER_THREADS");
    if (env == nullptr) env = std::getenv("FREETOKEN_PLE_READER_THREADS");  // the PLE-era name
    if (env != nullptr) {
      // kBatchEntries is the submit depth, so threads past it never get a read
      const int v = std::atoi(env);
      if (v > 0) n = std::min((unsigned)v, kBatchEntries);
    }
    for (unsigned i = 0; i < n; i++) workers_.emplace_back([this] { work(); });
  }

  ~ThreadPoolBatchReader() override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    work_cv_.notify_all();
    for (auto &w : workers_) w.join();
  }

  std::string name() const override { return "pread-pool x" + std::to_string(workers_.size()); }
  unsigned capacity() const override { return kBatchEntries; }

  void submit(unsigned tag, int fd, uint8_t *buf, int64_t len, int64_t need,
              int64_t off) override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.push_back(Req{tag, fd, buf, len, need, off});
      in_flight_++;
    }
    work_cv_.notify_one();
  }

  unsigned wait_one() override {
    std::unique_lock<std::mutex> lock(mu_);
    done_cv_.wait(lock, [this] { return !done_.empty(); });
    Done d = std::move(done_.front());
    done_.pop_front();
    in_flight_--;
    if (!d.error.empty()) throw std::runtime_error(d.error);
    return d.tag;
  }

  void drain() noexcept override {
    // wait out all in-flight reads: a late worker write must not race the slot's reuse
    std::unique_lock<std::mutex> lock(mu_);
    done_cv_.wait(lock, [this] { return done_.size() == in_flight_; });
    in_flight_ = 0;
    done_.clear();
  }

 private:
  struct Req {
    unsigned tag;
    int fd;
    uint8_t *buf;
    int64_t len;
    int64_t need;
    int64_t off;
  };
  struct Done {
    unsigned tag;
    std::string error;
  };

  void work() {
    for (;;) {
      Req r;
      {
        std::unique_lock<std::mutex> lock(mu_);
        work_cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (stop_) return;
        r = queue_.front();
        queue_.pop_front();
      }
      Done d{r.tag, {}};
      try {
        pread_min(r.fd, r.buf, r.len, r.need, r.off);
      } catch (const std::exception &e) {
        d.error = e.what();
      }
      {
        std::lock_guard<std::mutex> lock(mu_);
        done_.push_back(std::move(d));
      }
      done_cv_.notify_one();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable work_cv_, done_cv_;
  std::deque<Req> queue_;
  std::deque<Done> done_;
  size_t in_flight_ = 0;
  bool stop_ = false;
};

#if PLE_HAS_IO_URING

// Minimal single-issuer io_uring: submit up to `entries` reads, wait for all.
class IoUringBatchReader final : public BatchReader {
 public:
  IoUringBatchReader() = default;

  bool init(unsigned entries) {
    struct io_uring_params p{};
    fd_ = (int)syscall(__NR_io_uring_setup, entries, &p);
    if (fd_ < 0) return false;
    sq_size_ = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
    cq_size_ = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
    if (p.features & IORING_FEAT_SINGLE_MMAP) sq_size_ = cq_size_ = std::max(sq_size_, cq_size_);
    sq_ptr_ = mmap(nullptr, sq_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_,
                   IORING_OFF_SQ_RING);
    if (sq_ptr_ == MAP_FAILED) return false;
    cq_ptr_ = (p.features & IORING_FEAT_SINGLE_MMAP)
                  ? sq_ptr_
                  : mmap(nullptr, cq_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                         fd_, IORING_OFF_CQ_RING);
    if (cq_ptr_ == MAP_FAILED) return false;
    sqes_size_ = p.sq_entries * sizeof(io_uring_sqe);
    sqes_ = (io_uring_sqe *)mmap(nullptr, sqes_size_, PROT_READ | PROT_WRITE,
                                 MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQES);
    if (sqes_ == MAP_FAILED) return false;

    auto at = [&](void *base, uint32_t off) { return (uint8_t *)base + off; };
    sq_tail_ = (uint32_t *)at(sq_ptr_, p.sq_off.tail);
    sq_mask_ = (uint32_t *)at(sq_ptr_, p.sq_off.ring_mask);
    sq_array_ = (uint32_t *)at(sq_ptr_, p.sq_off.array);
    cq_head_ = (uint32_t *)at(cq_ptr_, p.cq_off.head);
    cq_tail_ = (uint32_t *)at(cq_ptr_, p.cq_off.tail);
    cq_mask_ = (uint32_t *)at(cq_ptr_, p.cq_off.ring_mask);
    cqes_ = (io_uring_cqe *)at(cq_ptr_, p.cq_off.cqes);
    entries_ = p.sq_entries;
    lens_.assign(entries_, 0);
    sq_shadow_tail_ = *sq_tail_;
    return true;
  }

  ~IoUringBatchReader() override {
    if (sqes_ && sqes_ != MAP_FAILED) munmap(sqes_, sqes_size_);
    if (cq_ptr_ && cq_ptr_ != MAP_FAILED && cq_ptr_ != sq_ptr_) munmap(cq_ptr_, cq_size_);
    if (sq_ptr_ && sq_ptr_ != MAP_FAILED) munmap(sq_ptr_, sq_size_);
    if (fd_ >= 0) ::close(fd_);
  }

  std::string name() const override { return "io_uring"; }
  unsigned capacity() const override { return entries_; }

  void submit(unsigned tag, int fd, uint8_t *buf, int64_t len, int64_t need,
              int64_t off) override {
    io_uring_sqe *sqe = &sqes_[sq_shadow_tail_ & *sq_mask_];
    std::memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IORING_OP_READ;
    sqe->fd = fd;
    sqe->addr = (uint64_t)(uintptr_t)buf;
    sqe->len = (uint32_t)len;
    sqe->off = (uint64_t)off;
    sqe->user_data = tag;
    sq_array_[sq_shadow_tail_ & *sq_mask_] = sq_shadow_tail_ & *sq_mask_;
    sq_shadow_tail_++;
    __atomic_store_n(sq_tail_, sq_shadow_tail_, __ATOMIC_RELEASE);
    lens_[tag] = need;
    to_submit_++;
    in_flight_++;
  }

  unsigned wait_one() override {
    for (;;) {
      uint32_t head = *cq_head_;
      uint32_t ctail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
      if (head != ctail) {
        const io_uring_cqe &cqe = cqes_[head & *cq_mask_];
        const unsigned tag = (unsigned)cqe.user_data;
        const int res = cqe.res;
        __atomic_store_n(cq_head_, head + 1, __ATOMIC_RELEASE);
        in_flight_--;
        if (res < 0)
          throw std::runtime_error(std::string("io_uring read: ") + std::strerror(-res));
        if (res < lens_[tag]) throw std::runtime_error("io_uring short read");
        return tag;
      }
      const unsigned to_submit = to_submit_;
      long rc = syscall(__NR_io_uring_enter, fd_, to_submit, 1, IORING_ENTER_GETEVENTS, nullptr, 0);
      if (rc < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error(std::string("io_uring_enter: ") + std::strerror(errno));
      }
      // partial submission is legal (signal, transient alloc); the rest stay in the ring
      to_submit_ = to_submit - (unsigned)rc;
    }
  }

  void drain() noexcept override {
    while (in_flight_ > 0) {
      uint32_t head = *cq_head_;
      uint32_t ctail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
      if (head != ctail) {
        __atomic_store_n(cq_head_, head + 1, __ATOMIC_RELEASE);
        in_flight_--;
        continue;
      }
      const unsigned to_submit = to_submit_;
      long rc = syscall(__NR_io_uring_enter, fd_, to_submit, 1, IORING_ENTER_GETEVENTS, nullptr, 0);
      if (rc < 0) {
        if (errno != EINTR) return;
        continue;
      }
      to_submit_ = to_submit - (unsigned)rc;
    }
  }

 private:
  int fd_ = -1;
  void *sq_ptr_ = nullptr, *cq_ptr_ = nullptr;
  io_uring_sqe *sqes_ = nullptr;
  size_t sq_size_ = 0, cq_size_ = 0, sqes_size_ = 0;
  uint32_t *sq_tail_ = nullptr, *sq_mask_ = nullptr, *sq_array_ = nullptr;
  uint32_t *cq_head_ = nullptr, *cq_tail_ = nullptr, *cq_mask_ = nullptr;
  io_uring_cqe *cqes_ = nullptr;
  unsigned entries_ = 0;
  uint32_t sq_shadow_tail_ = 0;
  unsigned to_submit_ = 0;
  unsigned in_flight_ = 0;
  std::vector<int64_t> lens_;
};

#endif  // PLE_HAS_IO_URING

unsigned ring_count() {
  const char *env = std::getenv("FREETOKEN_ROW_STORE_RINGS");
  const int v = env != nullptr ? std::atoi(env) : (int)kRingsDefault;
  return (unsigned)std::clamp(v, 1, (int)kRingsMax);
}

// Helper threads for the extra rings of a split fill: run(fn, n) hands fn(0..n-1) to n helpers, wait() joins them.
class RingHelpers {
 public:
  explicit RingHelpers(unsigned n) {
    try {
      for (unsigned i = 0; i < n; i++) threads_.emplace_back([this, i] { work(i); });
    } catch (...) {
      stop_all();  // a thread that could not start (a pids limit): join the started ones before rethrowing
      throw;
    }
  }

  ~RingHelpers() { stop_all(); }

  void run(const std::function<void(unsigned)> *fn, unsigned n) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      fn_ = fn;
      count_ = n;
      done_ = 0;
      gen_++;
    }
    cv_.notify_all();
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mu_);
    done_cv_.wait(lock, [this] { return done_ == count_; });
  }

 private:
  void stop_all() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &t : threads_) t.join();
    threads_.clear();
  }

  void work(unsigned i) {
    uint64_t seen = 0;
    for (;;) {
      const std::function<void(unsigned)> *fn;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return stop_ || gen_ != seen; });
        if (stop_) return;
        seen = gen_;
        if (i >= count_) continue;
        fn = fn_;
      }
      (*fn)(i);  // never throws: the caller's fn keeps its own errors
      {
        std::lock_guard<std::mutex> lock(mu_);
        done_++;
      }
      done_cv_.notify_one();
    }
  }

  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(unsigned)> *fn_ = nullptr;
  unsigned count_ = 0, done_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
};

std::unique_ptr<BatchReader> make_batch_reader(bool use_io_uring) {
#if PLE_HAS_IO_URING
  if (use_io_uring) {
    auto ring = std::make_unique<IoUringBatchReader>();
    if (ring->init(kBatchEntries)) return ring;
  }
#else
  (void)use_io_uring;
#endif
  return std::make_unique<ThreadPoolBatchReader>();
}

// ---- row store (platform-free) ----

int64_t wrap_mul(int64_t a, int64_t b) {
  return (int64_t)((uint64_t)a * (uint64_t)b);
}

int64_t pos_mod(int64_t v, int64_t m) {
  int64_t r = v % m;
  return r < 0 ? r + m : r;
}

// A RAM cache of rows read before (cache_rows > 0): 8-way set associative, least recently used out. Rows go in after a
// flush, copied from the staging the reads filled: a hit gives the bytes a read would.
class RowCache {
 public:
  static constexpr size_t kWays = 8;

  RowCache(size_t rows, int64_t row_bytes) : sets_(std::max<size_t>(1, rows / kWays)), row_bytes_(row_bytes) {
    tags_.assign(sets_ * kWays, -1);
    stamps_.assign(sets_ * kWays, 0);
    data_.resize(sets_ * kWays * (size_t)row_bytes_);
  }

  size_t rows() const { return sets_ * kWays; }
  int64_t hits() const { return hits_; }

  const uint8_t *find(int64_t row_id) {
    const size_t base = set_of(row_id) * kWays;
    for (size_t w = 0; w < kWays; w++)
      if (tags_[base + w] == row_id) {
        stamps_[base + w] = ++clock_;
        hits_++;
        return data_.data() + (base + w) * (size_t)row_bytes_;
      }
    return nullptr;
  }

  void put(int64_t row_id, const uint8_t *row) {
    const size_t base = set_of(row_id) * kWays;
    size_t way = 0;
    for (size_t w = 0; w < kWays; w++) {
      if (tags_[base + w] == row_id || tags_[base + w] < 0) {
        way = w;
        break;
      }
      if (stamps_[base + w] < stamps_[base + way]) way = w;
    }
    tags_[base + way] = row_id;
    stamps_[base + way] = ++clock_;
    std::memcpy(data_.data() + (base + way) * (size_t)row_bytes_, row, row_bytes_);
  }

 private:
  size_t set_of(int64_t row_id) const { return (size_t)(((uint64_t)row_id * 0x9E3779B97F4A7C15ull) >> 20) % sets_; }

  size_t sets_;
  int64_t row_bytes_;
  std::vector<int64_t> tags_;
  std::vector<uint64_t> stamps_;
  std::vector<uint8_t> data_;
  uint64_t clock_ = 0;
  int64_t hits_ = 0;
};

class RowStore {
  struct Extent {
    const TableFile *file;
    int64_t base;
  };

 public:
  RowStore(std::vector<std::string> paths, std::vector<int64_t> extent_file,
           std::vector<int64_t> extent_base, int64_t rows_per_extent, int64_t row_bytes,
           int64_t row_stride, bool use_io_uring, int64_t cache_rows = 0)
      : row_bytes_(row_bytes), row_stride_(row_stride), rows_per_extent_(rows_per_extent) {
    if (row_bytes_ <= 0 || row_stride_ < row_bytes_ || rows_per_extent_ <= 0)
      throw std::runtime_error("row store geometry: want row_bytes > 0, row_stride >= row_bytes, rows_per_extent > 0");
    if (row_bytes_ > kPage)
      throw std::runtime_error("row_bytes " + std::to_string(row_bytes_) +
                               " exceeds a page; bounce slots assume one-page rows");
    if (extent_file.size() != extent_base.size() || extent_file.empty())
      throw std::runtime_error("row store geometry: extent_file and extent_base must be equal-length and non-empty");
    for (const std::string &p : paths)
      files_.push_back(std::make_unique<TableFile>(p));
    const int64_t extent_bytes = (rows_per_extent_ - 1) * row_stride_ + row_bytes_;
    for (size_t e = 0; e < extent_file.size(); e++) {
      const size_t fi = (size_t)extent_file.at(e);
      const int64_t base = extent_base.at(e);
      if (base + extent_bytes > files_.at(fi)->size())
        throw std::runtime_error(paths[fi] + ": extent needs " +
                                 std::to_string(base + extent_bytes) + " bytes, file has " +
                                 std::to_string(files_[fi]->size()));
      extents_.push_back(Extent{files_[fi].get(), base});
    }
    readers_.push_back(make_batch_reader(use_io_uring));
    // the pread pool is parallel already; only io_uring gets extra rings
    for (unsigned i = 1; i < ring_count() && readers_[0]->name() == "io_uring"; i++) {
      auto r = make_batch_reader(true);
      if (r->name() != "io_uring") break;
      readers_.push_back(std::move(r));
    }
    for (auto &r : readers_) {
      if (r->capacity() < 2) throw std::runtime_error("row store reader: a row across two pages needs two reads in flight");
      bounces_.push_back(page_aligned_alloc((size_t)r->capacity() * kPage));  // one page a tag
    }
    if (readers_.size() > 1) {
      try {
        helpers_ = std::make_unique<RingHelpers>((unsigned)readers_.size() - 1);
      } catch (const std::system_error &) {  // no helper threads: one ring
        while (readers_.size() > 1) {
          readers_.pop_back();
          free(bounces_.back());
          bounces_.pop_back();
        }
      }
    }
    if (cache_rows > 0) cache_ = std::make_unique<RowCache>((size_t)cache_rows, row_bytes_);
  }

  // helpers and readers first: a still-running read must not land in freed bounce memory
  virtual ~RowStore() {
    helpers_.reset();
    readers_.clear();
    for (uint8_t *b : bounces_) free(b);
  }

  RowStore(const RowStore &) = delete;
  RowStore &operator=(const RowStore &) = delete;

  int64_t row_bytes() const { return row_bytes_; }
  int64_t total_rows() const { return (int64_t)extents_.size() * rows_per_extent_; }

  // Queue n row ids (int64 at rows_addr); row i lands at staging_addr + i * dst_stride
  // (dst_stride = 0 packs rows back to back). Out-of-range ids throw before any I/O is
  // issued. No I/O until flush().
  void stage_rows(uintptr_t rows_addr, int64_t n, uintptr_t staging_addr, int64_t dst_stride) {
    const int64_t *rows = reinterpret_cast<const int64_t *>(rows_addr);
    uint8_t *staging = reinterpret_cast<uint8_t *>(staging_addr);
    if (dst_stride == 0) dst_stride = row_bytes_;
    if (dst_stride < row_bytes_)
      throw std::runtime_error("stage_rows: dst_stride below row_bytes would overlap rows");
    const int64_t total = total_rows();
    for (int64_t i = 0; i < n; i++) {
      if (rows[i] < 0 || rows[i] >= total)
        throw std::out_of_range("stage_rows: row id " + std::to_string(rows[i]) + " outside [0, " +
                                std::to_string(total) + ")");
    }
    try {
      for (int64_t i = 0; i < n; i++) request_row(rows[i], staging + (size_t)i * (size_t)dst_stride);
    } catch (...) {
      drop_pending();  // a half-staged fill must not reach the next flush (nor the cache under another row's slot)
      throw;
    }
  }

  // One batched disk round for everything staged; signals even when nothing was.
  void flush(uintptr_t signal_addr) {
    struct Cleanup {
      RowStore *s;
      ~Cleanup() {
        s->pending_.clear();
        s->pending_index_.clear();
      }
    } cleanup{this};
    flush_pending();
    if (signal_addr) signal_flag(signal_addr);
    // after the signal: the rows the reads just wrote (a failed flush threw above and caches nothing); the consumer
    // only reads the staging, and the next fill comes from this same thread
    if (cache_)
      for (const Pending &p : pending_) cache_->put(p.row_id, p.dsts[0]);
  }

  int64_t cache_rows() const { return cache_ ? (int64_t)cache_->rows() : 0; }
  int64_t cache_hits() const { return cache_ ? cache_->hits() : 0; }  // not synchronized: read it when no fill runs

  std::string io_backend() const {
    size_t direct = 0;
    for (const auto &f : files_) direct += f->direct_io() ? 1 : 0;
    std::string s = readers_[0]->name();
    if (readers_.size() > 1) s += " x" + std::to_string(readers_.size());
    if (cache_) s += ", RAM cache of " + std::to_string(cache_->rows()) + " rows";
    if (direct == files_.size()) return s + ", O_DIRECT";
    return s + ", buffered " + std::to_string(files_.size() - direct) + "/" +
           std::to_string(files_.size()) + " files";
  }

 protected:
  void drop_pending() {
    pending_.clear();
    pending_index_.clear();
  }

  // Queue dst on this fill's pending batch; duplicate rows fan out from one read.
  void request_row(int64_t row_id, uint8_t *dst) {
    auto pit = pending_index_.find(row_id);
    if (pit != pending_index_.end()) {
      pending_[pit->second].dsts.push_back(dst);
      return;
    }
    if (cache_) {
      if (const uint8_t *row = cache_->find(row_id)) {
        std::memcpy(dst, row, row_bytes_);
        return;
      }
    }
    const Extent &ext = extents_[row_id / rows_per_extent_];
    const int64_t off = ext.base + (row_id % rows_per_extent_) * row_stride_;
    Pending p{ext.file, off, row_bytes_, 0, {dst}, row_id};
    if (ext.file->direct_io()) {
      // full aligned span even past EOF; truncating would break direct-I/O alignment
      p.read_off = off & ~(kPage - 1);
      p.row_off = off - p.read_off;
      p.read_len = ((off + row_bytes_ + kPage - 1) & ~(kPage - 1)) - p.read_off;
    }
    pending_index_.emplace(row_id, pending_.size());
    pending_.push_back(std::move(p));
  }

 private:
  struct Pending {
    const TableFile *file;
    int64_t read_off;
    int64_t read_len;
    int64_t row_off;  // row payload start inside the read buffer
    std::vector<uint8_t *> dsts;
    int64_t row_id;
  };

  // Read every pending row in reader-capacity batches and fan out the copies; a large fill
  // splits into one contiguous range per ring (the ranges' rows and destinations are disjoint).
  void flush_pending() {
    if (pending_.empty()) return;
    const size_t total = pending_.size();
    const size_t rings = total < kSplitMinRows ? 1 : std::min(readers_.size(), total / (kSplitMinRows / 4));
    if (rings <= 1) {
      read_range(0, 0, total);
      return;
    }
    std::vector<std::string> errors(rings);
    auto range = [&](unsigned i) {
      try {
        read_range(i, total * i / rings, total * (i + 1) / rings);
      } catch (const std::exception &e) {
        errors[i] = e.what();
      }
    };
    const std::function<void(unsigned)> helper = [&](unsigned i) { range(i + 1); };
    helpers_->run(&helper, (unsigned)rings - 1);
    range(0);
    helpers_->wait();
    for (const std::string &e : errors)
      if (!e.empty()) throw std::runtime_error(e);
  }

  // The pipelined read of pending_[lo, hi) on ring r: at most capacity() pages in flight, each tag
  // reading one page into its own bounce page. A direct-I/O row that crosses a page boundary reads its two
  // pages apart and is joined from both: on the reference NVMe a read spanning two 4 KiB units (8 KiB, or
  // 1 KiB across a boundary) halved the throughput of the whole batch, even at 2.5 % of the reads. Its
  // second page goes out after another row's page, or the block layer merges the two back (plug merging
  // only tries the last request queued)
  void read_range(size_t r, size_t lo, size_t hi) {
    BatchReader &reader = *readers_[r];
    uint8_t *bounce = bounces_[r];
    const unsigned cap = reader.capacity();
    std::vector<size_t> tag_pending(cap);
    std::vector<unsigned> free_tags;
    for (unsigned t = cap; t-- > 0;) free_tags.push_back(t);
    // per row of the range: its pages' tags and how many of its pages are still in flight
    std::vector<unsigned> first_tag(hi - lo), second_tag(hi - lo);
    std::vector<uint8_t> left(hi - lo);
    std::deque<size_t> second_pages;  // rows whose first page is out, the second not yet
    size_t next = lo, last = hi;
    unsigned reserved = 0;  // free tags promised to the second pages above
    unsigned inflight = 0;
    auto pages = [](const Pending &p) { return p.read_len > kPage ? 2u : 1u; };
    auto submit = [&](size_t i, unsigned part) {
      const Pending &p = pending_[i];
      const unsigned tag = free_tags.back();
      free_tags.pop_back();
      tag_pending[tag] = i;
      uint8_t *page = bounce + (size_t)tag * kPage;
      if (pages(p) == 1) {
        reader.submit(tag, p.file->native_fd(), page, p.read_len, p.row_off + row_bytes_, p.read_off);
      } else {
        const int64_t need = part == 0 ? kPage : p.row_off + row_bytes_ - kPage;
        reader.submit(tag, p.file->native_fd(), page, kPage, need, p.read_off + (int64_t)part * kPage);
      }
      (part == 0 ? first_tag : second_tag)[i - lo] = tag;
      last = i;
      inflight++;
    };
    auto fill = [&]() {
      for (;;) {
        // with nothing in flight it goes out regardless, or a range of two-page rows could hold every tag
        if (!second_pages.empty() && (second_pages.front() != last || next == hi || inflight == 0)) {
          submit(second_pages.front(), 1);
          second_pages.pop_front();
          reserved--;
        } else if (next < hi && free_tags.size() >= reserved + pages(pending_[next])) {
          const unsigned n = pages(pending_[next]);
          left[next - lo] = (uint8_t)n;
          submit(next, 0);
          if (n == 2) {
            second_pages.push_back(next);
            reserved++;
          }
          next++;
        } else {
          return;
        }
        if (free_tags.empty()) return;
      }
    };
    try {
      fill();
      for (size_t completed = lo; completed < hi;) {
        const unsigned tag = reader.wait_one();
        const size_t i = tag_pending[tag];
        const Pending &p = pending_[i];
        inflight--;
        if (--left[i - lo] == 0) {
          const uint8_t *page0 = bounce + (size_t)first_tag[i - lo] * kPage;
          if (pages(p) == 1) {
            for (uint8_t *dst : p.dsts) std::memcpy(dst, page0 + p.row_off, row_bytes_);
          } else {
            const uint8_t *page1 = bounce + (size_t)second_tag[i - lo] * kPage;
            const int64_t head = kPage - p.row_off;
            for (uint8_t *dst : p.dsts) {
              std::memcpy(dst, page0 + p.row_off, head);
              std::memcpy(dst + head, page1, row_bytes_ - head);
            }
            free_tags.push_back(second_tag[i - lo]);
          }
          p.file->discard_cache(p.read_off, p.read_len);
          free_tags.push_back(first_tag[i - lo]);
          completed++;
        }
        fill();
      }
    } catch (...) {
      reader.drain();
      throw;
    }
  }

  int64_t row_bytes_, row_stride_, rows_per_extent_;
  std::vector<std::unique_ptr<TableFile>> files_;
  std::vector<Extent> extents_;
  std::vector<std::unique_ptr<BatchReader>> readers_;  // [0] serves every small fill
  std::vector<uint8_t *> bounces_;                     // one bounce area per reader
  std::unique_ptr<RingHelpers> helpers_;               // drive readers_[1..]
  std::unique_ptr<RowCache> cache_;

  std::vector<Pending> pending_;
  std::unordered_map<int64_t, size_t> pending_index_;
};

// The Qwen3.8-Flash-Next PLE table: RowStore plus the checkpoint's n-gram hash, so a fill
// takes raw token runs and never crosses the Python boundary per row.
class PleStore final : public RowStore {
 public:
  PleStore(std::vector<std::string> paths, std::vector<int64_t> extent_file,
           std::vector<int64_t> extent_base, int64_t rows_per_extent, int64_t row_bytes,
           int64_t row_stride, std::vector<int64_t> multipliers, std::vector<int64_t> head_vocab_sizes,
           std::vector<int64_t> head_offsets, int64_t eos_token_id, bool use_io_uring, int64_t cache_rows = 0)
      : RowStore(std::move(paths), std::move(extent_file), std::move(extent_base), rows_per_extent,
                 row_bytes, row_stride, use_io_uring, cache_rows),
        mult_(std::move(multipliers)),
        sizes_(std::move(head_vocab_sizes)),
        offsets_(std::move(head_offsets)),
        eos_(eos_token_id) {
    if (mult_.size() != 3 || sizes_.size() != offsets_.size() || sizes_.empty())
      throw std::runtime_error("PLE hash geometry: want 3 multipliers and equal-length head tables");
  }

  // Row ids for the token at w[2] with context (w[0], w[1]); mirrors NGramEmbedding.row_ids incl. the eos barrier.
  void hash_rows(const int64_t *w, int64_t *rows) {
    const int64_t prev1 = w[1];
    const int64_t prev2 = prev1 == eos_ ? eos_ : w[0];
    const int64_t bigram = wrap_mul(w[2], mult_[0]) ^ wrap_mul(prev1, mult_[1]);
    const int64_t trigram = bigram ^ wrap_mul(prev2, mult_[2]);
    const size_t half = sizes_.size() / 2;
    for (size_t h = 0; h < sizes_.size(); h++)
      rows[h] = pos_mod(h < half ? bigram : trigram, sizes_[h]) + offsets_[h];
  }

  // Hash and queue one run: tokens_addr holds n+2 ids, the leading two are context. No I/O until flush().
  void stage(uintptr_t tokens_addr, int64_t n, uintptr_t staging_addr) {
    const int64_t *tokens = reinterpret_cast<const int64_t *>(tokens_addr);
    uint8_t *staging = reinterpret_cast<uint8_t *>(staging_addr);
    const size_t heads = sizes_.size();
    try {
      std::vector<int64_t> rows(heads);
      for (int64_t i = 0; i < n; i++) {
        hash_rows(tokens + i, rows.data());
        for (size_t h = 0; h < heads; h++)
          request_row(rows[h], staging + ((size_t)i * heads + h) * (size_t)row_bytes());
      }
    } catch (...) {
      drop_pending();
      throw;
    }
  }

 private:
  std::vector<int64_t> mult_, sizes_, offsets_;
  int64_t eos_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<RowStore>(m, "RowStore")
      .def(py::init<std::vector<std::string>, std::vector<int64_t>, std::vector<int64_t>,
                    int64_t, int64_t, int64_t, bool, int64_t>(),
           py::arg("paths"), py::arg("extent_file"), py::arg("extent_base"),
           py::arg("rows_per_extent"), py::arg("row_bytes"), py::arg("row_stride"),
           py::arg("use_io_uring") = true, py::arg("cache_rows") = 0)
      .def("stage_rows", &RowStore::stage_rows, py::arg("rows_addr"), py::arg("n"),
           py::arg("staging_addr"), py::arg("dst_stride") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("flush", &RowStore::flush, py::arg("signal_addr") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("io_backend", &RowStore::io_backend)
      .def_property_readonly("cache_rows", &RowStore::cache_rows)
      .def_property_readonly("cache_hits", &RowStore::cache_hits)
      .def_property_readonly("row_bytes", &RowStore::row_bytes)
      .def_property_readonly("total_rows", &RowStore::total_rows);
  py::class_<PleStore, RowStore>(m, "PleStore")
      .def(py::init<std::vector<std::string>, std::vector<int64_t>, std::vector<int64_t>,
                    int64_t, int64_t, int64_t, std::vector<int64_t>,
                    std::vector<int64_t>, std::vector<int64_t>, int64_t, bool, int64_t>(),
           py::arg("paths"), py::arg("extent_file"), py::arg("extent_base"),
           py::arg("rows_per_extent"), py::arg("row_bytes"), py::arg("row_stride"),
           py::arg("multipliers"),
           py::arg("head_vocab_sizes"), py::arg("head_offsets"), py::arg("eos_token_id"),
           py::arg("use_io_uring") = true, py::arg("cache_rows") = 0)
      .def("stage", &PleStore::stage, py::arg("tokens_addr"), py::arg("n"),
           py::arg("staging_addr"), py::call_guard<py::gil_scoped_release>());
  m.def("memop_write", &memop_write, py::arg("stream"), py::arg("addr"), py::arg("value"));
  m.def("memop_wait_geq", &memop_wait_geq, py::arg("stream"), py::arg("addr"), py::arg("value"));
  m.def("memop_wait_reset", &memop_wait_reset, py::arg("stream"), py::arg("flag_addr"));
  m.def("signal_flag", &signal_flag, py::arg("flag_addr"));
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
  m.def("alloc_signal_tensor", &alloc_signal_tensor);
  m.def("capture_memops_supported", &capture_memops_supported);
#endif
}
