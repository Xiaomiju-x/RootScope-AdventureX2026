// RootScope v3 persistent libdnn worker for the RDK X5.
//
// The vendor hrt_model_exec binary-input contract was measured on the actual
// board: contiguous uint8 RGB NCHW valid bytes are supplied after copying
// validShape into alignedShape.  hbm_runtime instead allocates the original
// padded aligned shape and is therefore not numerically equivalent for this
// model.  This worker preserves the measured contract while loading the model
// exactly once.
//
// Authority boundary: this process only opens the hash-bound model file and
// communicates over inherited stdin/stdout/stderr.  It contains no camera,
// network, serial, GPIO, service-manager, or actuator code.

#include <dnn/hb_dnn.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::array<uint8_t, 8> kRequestMagic = {
    'R', 'S', 'N', 'V', '3', 'R', 'E', 'Q'};
constexpr std::array<uint8_t, 8> kResponseMagic = {
    'R', 'S', 'N', 'V', '3', 'R', 'S', 'P'};
constexpr uint32_t kProtocolVersion = 1;
constexpr std::array<int32_t, 4> kInputShape = {1, 3, 224, 224};
constexpr std::array<int32_t, 4> kOutputShape = {1, 4, 1, 1};
constexpr uint32_t kInputBytes = 1U * 3U * 224U * 224U;
constexpr uint32_t kOutputBytes = 4U * sizeof(float);

void Check(int32_t code, const char* operation) {
  if (code != 0) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             std::to_string(code));
  }
}

uint32_t RotateRight(uint32_t value, uint32_t amount) {
  return (value >> amount) | (value << (32U - amount));
}

class Sha256 {
 public:
  Sha256()
      : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
               0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

  void Update(const uint8_t* data, std::size_t size) {
    total_bytes_ += size;
    while (size > 0) {
      const std::size_t available = buffer_.size() - buffered_;
      const std::size_t take = size < available ? size : available;
      std::memcpy(buffer_.data() + buffered_, data, take);
      buffered_ += take;
      data += take;
      size -= take;
      if (buffered_ == buffer_.size()) {
        Transform(buffer_.data());
        buffered_ = 0;
      }
    }
  }

  std::array<uint8_t, 32> Final() {
    const uint64_t bit_count = static_cast<uint64_t>(total_bytes_) * 8U;
    buffer_[buffered_++] = 0x80U;
    if (buffered_ > 56U) {
      while (buffered_ < 64U) buffer_[buffered_++] = 0U;
      Transform(buffer_.data());
      buffered_ = 0;
    }
    while (buffered_ < 56U) buffer_[buffered_++] = 0U;
    for (int index = 7; index >= 0; --index) {
      buffer_[buffered_++] =
          static_cast<uint8_t>((bit_count >> (index * 8)) & 0xffU);
    }
    Transform(buffer_.data());
    std::array<uint8_t, 32> digest{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
      digest[word * 4U] = static_cast<uint8_t>(state_[word] >> 24U);
      digest[word * 4U + 1U] = static_cast<uint8_t>(state_[word] >> 16U);
      digest[word * 4U + 2U] = static_cast<uint8_t>(state_[word] >> 8U);
      digest[word * 4U + 3U] = static_cast<uint8_t>(state_[word]);
    }
    return digest;
  }

 private:
  void Transform(const uint8_t* block) {
    static constexpr std::array<uint32_t, 64> kRound = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<uint32_t, 64> schedule{};
    for (std::size_t index = 0; index < 16; ++index) {
      const std::size_t offset = index * 4U;
      schedule[index] =
          (static_cast<uint32_t>(block[offset]) << 24U) |
          (static_cast<uint32_t>(block[offset + 1U]) << 16U) |
          (static_cast<uint32_t>(block[offset + 2U]) << 8U) |
          static_cast<uint32_t>(block[offset + 3U]);
    }
    for (std::size_t index = 16; index < schedule.size(); ++index) {
      const uint32_t s0 = RotateRight(schedule[index - 15U], 7U) ^
                          RotateRight(schedule[index - 15U], 18U) ^
                          (schedule[index - 15U] >> 3U);
      const uint32_t s1 = RotateRight(schedule[index - 2U], 17U) ^
                          RotateRight(schedule[index - 2U], 19U) ^
                          (schedule[index - 2U] >> 10U);
      schedule[index] = schedule[index - 16U] + s0 +
                        schedule[index - 7U] + s1;
    }
    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];
    for (std::size_t index = 0; index < schedule.size(); ++index) {
      const uint32_t sum1 = RotateRight(e, 6U) ^ RotateRight(e, 11U) ^
                            RotateRight(e, 25U);
      const uint32_t choose = (e & f) ^ ((~e) & g);
      const uint32_t temporary1 =
          h + sum1 + choose + kRound[index] + schedule[index];
      const uint32_t sum0 = RotateRight(a, 2U) ^ RotateRight(a, 13U) ^
                            RotateRight(a, 22U);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<uint32_t, 8> state_{};
  std::array<uint8_t, 64> buffer_{};
  std::size_t buffered_ = 0;
  std::size_t total_bytes_ = 0;
};

std::string Sha256File(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open model for SHA-256");
  Sha256 hash;
  std::array<uint8_t, 1024U * 1024U> buffer{};
  while (stream) {
    stream.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    const auto count = stream.gcount();
    if (count > 0) {
      hash.Update(buffer.data(), static_cast<std::size_t>(count));
    }
  }
  if (!stream.eof()) throw std::runtime_error("failed while hashing model");
  const auto digest = hash.Final();
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (uint8_t value : digest) output << std::setw(2) << unsigned(value);
  return output.str();
}

bool IsLowerHexSha256(const std::string& value) {
  if (value.size() != 64U) return false;
  for (const char character : value) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

void ValidateRegularLockedFile(const std::string& path, const char* label) {
  struct stat metadata {};
  if (lstat(path.c_str(), &metadata) != 0) {
    throw std::runtime_error(std::string("lstat failed for ") + label);
  }
  if (S_ISLNK(metadata.st_mode) || !S_ISREG(metadata.st_mode)) {
    throw std::runtime_error(std::string(label) +
                             " must be one regular non-symlink file");
  }
  if ((metadata.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    throw std::runtime_error(std::string(label) +
                             " must not be group/world writable");
  }
}

bool ReadExact(int descriptor, uint8_t* destination, std::size_t size,
               bool allow_clean_eof) {
  std::size_t offset = 0;
  while (offset < size) {
    const ssize_t count =
        read(descriptor, destination + offset, size - offset);
    if (count == 0) {
      if (allow_clean_eof && offset == 0) return false;
      throw std::runtime_error("unexpected EOF in request frame");
    }
    if (count < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("stdin read failed");
    }
    offset += static_cast<std::size_t>(count);
  }
  return true;
}

void WriteExact(int descriptor, const uint8_t* source, std::size_t size) {
  std::size_t offset = 0;
  while (offset < size) {
    const ssize_t count = write(descriptor, source + offset, size - offset);
    if (count < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("stdout write failed");
    }
    offset += static_cast<std::size_t>(count);
  }
}

uint32_t DecodeU32(const uint8_t* source) {
  return static_cast<uint32_t>(source[0]) |
         (static_cast<uint32_t>(source[1]) << 8U) |
         (static_cast<uint32_t>(source[2]) << 16U) |
         (static_cast<uint32_t>(source[3]) << 24U);
}

uint64_t DecodeU64(const uint8_t* source) {
  uint64_t value = 0;
  for (int index = 7; index >= 0; --index) {
    value = (value << 8U) | source[index];
  }
  return value;
}

void EncodeU32(uint32_t value, uint8_t* destination) {
  for (int index = 0; index < 4; ++index) {
    destination[index] = static_cast<uint8_t>(value >> (index * 8U));
  }
}

void EncodeU64(uint64_t value, uint8_t* destination) {
  for (int index = 0; index < 8; ++index) {
    destination[index] = static_cast<uint8_t>(value >> (index * 8U));
  }
}

std::size_t ElementCount(const hbDNNTensorShape& shape) {
  std::size_t result = 1;
  for (int32_t index = 0; index < shape.numDimensions; ++index) {
    if (shape.dimensionSize[index] <= 0) {
      throw std::runtime_error("tensor shape contains a non-positive dimension");
    }
    result *= static_cast<std::size_t>(shape.dimensionSize[index]);
  }
  return result;
}

void RequireShape(const hbDNNTensorShape& shape,
                  const std::array<int32_t, 4>& expected,
                  const char* label) {
  if (shape.numDimensions != static_cast<int32_t>(expected.size())) {
    throw std::runtime_error(std::string(label) + " dimension count changed");
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (shape.dimensionSize[index] != expected[index]) {
      throw std::runtime_error(std::string(label) + " shape changed");
    }
  }
}

struct Arguments {
  std::string model;
  std::string model_sha256;
  std::string model_name;
};

Arguments ParseArguments(int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (index + 1 >= argc) {
      throw std::runtime_error("every option requires one value");
    }
    const std::string value = argv[++index];
    if (option == "--model") {
      result.model = value;
    } else if (option == "--model-sha256") {
      result.model_sha256 = value;
    } else if (option == "--expected-model-name") {
      result.model_name = value;
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (result.model.empty() || result.model_sha256.empty() ||
      result.model_name.empty()) {
    throw std::runtime_error(
        "required: --model, --model-sha256, --expected-model-name");
  }
  if (!IsLowerHexSha256(result.model_sha256)) {
    throw std::runtime_error("model SHA-256 must be 64 lowercase hex digits");
  }
  return result;
}

class PersistentModel {
 public:
  explicit PersistentModel(const Arguments& arguments) {
    ValidateRegularLockedFile(arguments.model, "model");
    const std::string actual_sha256 = Sha256File(arguments.model);
    if (actual_sha256 != arguments.model_sha256) {
      throw std::runtime_error("model SHA-256 mismatch: " + actual_sha256);
    }
    const char* files[] = {arguments.model.c_str()};
    Check(hbDNNInitializeFromFiles(&packed_, files, 1),
          "hbDNNInitializeFromFiles");
    const char** names = nullptr;
    int32_t name_count = 0;
    Check(hbDNNGetModelNameList(&names, &name_count, packed_),
          "hbDNNGetModelNameList");
    if (name_count != 1 || names == nullptr ||
        arguments.model_name != names[0]) {
      throw std::runtime_error("model name/count does not match contract");
    }
    Check(hbDNNGetModelHandle(&model_, packed_, names[0]),
          "hbDNNGetModelHandle");
    int32_t input_count = 0;
    int32_t output_count = 0;
    Check(hbDNNGetInputCount(&input_count, model_), "hbDNNGetInputCount");
    Check(hbDNNGetOutputCount(&output_count, model_), "hbDNNGetOutputCount");
    if (input_count != 1 || output_count != 1) {
      throw std::runtime_error("expected exactly one input and one output");
    }
    Check(hbDNNGetInputTensorProperties(&input_.properties, model_, 0),
          "hbDNNGetInputTensorProperties");
    Check(hbDNNGetOutputTensorProperties(&output_.properties, model_, 0),
          "hbDNNGetOutputTensorProperties");
    RequireShape(input_.properties.validShape, kInputShape, "input valid");
    RequireShape(output_.properties.validShape, kOutputShape, "output valid");
    if (input_.properties.tensorLayout != HB_DNN_LAYOUT_NCHW ||
        input_.properties.tensorType != HB_DNN_IMG_TYPE_RGB) {
      throw std::runtime_error("input must be uint8 RGB NCHW");
    }
    if (output_.properties.tensorType != HB_DNN_TENSOR_TYPE_F32 ||
        ElementCount(output_.properties.validShape) != 4U ||
        output_.properties.alignedByteSize < kOutputBytes) {
      throw std::runtime_error("output must be exactly four float32 logits");
    }
    // The crucial, measured hrt_model_exec binary-input contract.
    input_.properties.alignedShape = input_.properties.validShape;
    Check(hbSysAllocCachedMem(&input_.sysMem[0], kInputBytes),
          "hbSysAllocCachedMem(input-valid-bytes)");
    input_allocated_ = true;
    Check(hbSysAllocCachedMem(&output_.sysMem[0],
                              output_.properties.alignedByteSize),
          "hbSysAllocCachedMem(output)");
    output_allocated_ = true;
  }

  ~PersistentModel() {
    if (task_ != nullptr) hbDNNReleaseTask(task_);
    if (output_allocated_) hbSysFreeMem(&output_.sysMem[0]);
    if (input_allocated_) hbSysFreeMem(&input_.sysMem[0]);
    if (packed_ != nullptr) hbDNNRelease(packed_);
  }

  std::array<float, 4> Infer(const uint8_t* source) {
    std::memcpy(input_.sysMem[0].virAddr, source, kInputBytes);
    Check(hbSysFlushMem(&input_.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
          "hbSysFlushMem(input)");
    hbDNNInferCtrlParam control;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
    hbDNNTensor* output_pointer = &output_;
    Check(hbDNNInfer(&task_, &output_pointer, &input_, model_, &control),
          "hbDNNInfer");
    try {
      Check(hbDNNWaitTaskDone(task_, 0), "hbDNNWaitTaskDone");
      Check(hbSysFlushMem(&output_.sysMem[0],
                          HB_SYS_MEM_CACHE_INVALIDATE),
            "hbSysFlushMem(output)");
      const auto* logits =
          static_cast<const float*>(output_.sysMem[0].virAddr);
      std::array<float, 4> result{};
      for (std::size_t index = 0; index < result.size(); ++index) {
        if (!std::isfinite(logits[index])) {
          throw std::runtime_error("libdnn returned a non-finite logit");
        }
        result[index] = logits[index];
      }
      Check(hbDNNReleaseTask(task_), "hbDNNReleaseTask");
      task_ = nullptr;
      return result;
    } catch (...) {
      if (task_ != nullptr) {
        hbDNNReleaseTask(task_);
        task_ = nullptr;
      }
      throw;
    }
  }

 private:
  hbPackedDNNHandle_t packed_ = nullptr;
  hbDNNHandle_t model_ = nullptr;
  hbDNNTaskHandle_t task_ = nullptr;
  hbDNNTensor input_{};
  hbDNNTensor output_{};
  bool input_allocated_ = false;
  bool output_allocated_ = false;
};

void WriteResponse(int descriptor, uint64_t request_id, uint64_t latency_ns,
                   const std::array<float, 4>& logits) {
  std::array<uint8_t, 36> header{};
  std::copy(kResponseMagic.begin(), kResponseMagic.end(), header.begin());
  EncodeU32(kProtocolVersion, header.data() + 8U);
  EncodeU64(request_id, header.data() + 12U);
  EncodeU32(0U, header.data() + 20U);
  EncodeU64(latency_ns, header.data() + 24U);
  EncodeU32(kOutputBytes, header.data() + 32U);
  std::array<uint8_t, kOutputBytes> payload{};
  for (std::size_t index = 0; index < logits.size(); ++index) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(logits[index]));
    std::memcpy(&bits, &logits[index], sizeof(bits));
    EncodeU32(bits, payload.data() + index * sizeof(float));
  }
  WriteExact(descriptor, header.data(), header.size());
  WriteExact(descriptor, payload.data(), payload.size());
}

}  // namespace

int main(int argc, char** argv) {
  int protocol_output = -1;
  try {
    // libdnn writes informational messages to stdout. Preserve the inherited
    // stdout pipe for protocol frames, then route vendor stdout to stderr so
    // no textual log can corrupt a binary response.
    protocol_output = dup(STDOUT_FILENO);
    if (protocol_output < 0 ||
        dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
      throw std::runtime_error("failed to isolate protocol stdout");
    }
    const Arguments arguments = ParseArguments(argc, argv);
    PersistentModel model(arguments);
    std::vector<uint8_t> payload(kInputBytes);
    while (true) {
      std::array<uint8_t, 24> header{};
      if (!ReadExact(STDIN_FILENO, header.data(), header.size(), true)) break;
      if (!std::equal(kRequestMagic.begin(), kRequestMagic.end(),
                      header.begin())) {
        throw std::runtime_error("request magic mismatch");
      }
      if (DecodeU32(header.data() + 8U) != kProtocolVersion) {
        throw std::runtime_error("request protocol version mismatch");
      }
      const uint64_t request_id = DecodeU64(header.data() + 12U);
      if (request_id == 0U) {
        throw std::runtime_error("request id zero is reserved");
      }
      if (DecodeU32(header.data() + 20U) != kInputBytes) {
        throw std::runtime_error("request payload length/shape mismatch");
      }
      ReadExact(STDIN_FILENO, payload.data(), payload.size(), false);
      const auto started = std::chrono::steady_clock::now();
      const std::array<float, 4> logits = model.Infer(payload.data());
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started);
      WriteResponse(protocol_output, request_id,
                    static_cast<uint64_t>(elapsed.count()), logits);
    }
    close(protocol_output);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rootscope-native-libdnn-worker: " << error.what() << "\n";
    if (protocol_output >= 0) close(protocol_output);
    return 1;
  }
}
