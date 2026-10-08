/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hlsl_compiler.hpp"
#include "module_compiler.hpp"
#include "workload_impl.hpp"

#include <dxc/Support/Global.h>
#include <dxc/Support/HLSLOptions.h>
#include <dxc/dxcapi.h>
#include <llvm/Support/FileSystem.h>

#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>

#if defined(_WIN32)
#    include <atlbase.h>
#    include <windows.h>
#endif

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hlsl {
HRESULT SetupRegistryPassForHLSL();
HRESULT SetupRegistryPassForPIX();
} // namespace hlsl

namespace mlsdk::workloadlib::detail {
namespace {

class HlslCompiler {
  public:
    HlslCompiler(const HlslCompiler &) = delete;
    HlslCompiler &operator=(const HlslCompiler &) = delete;
    HlslCompiler(HlslCompiler &&) = delete;
    HlslCompiler &operator=(HlslCompiler &&) = delete;
    ~HlslCompiler() = default;

    static HlslCompiler &get();

    std::pair<std::string, std::vector<uint32_t>> compile(const std::string &source, const std::string &entryPoint,
                                                          const std::string &debugName,
                                                          const std::string &preprocessorOptions,
                                                          const std::vector<std::string> &shaderDirs);

  private:
    HlslCompiler() = default;
};

#if !defined(_WIN32)
bool isUtf8ContinuationByte(unsigned char c) noexcept { return (c & 0xC0U) == 0x80U; }

int utf8ContinuationPayload(const std::string &inputString, std::size_t index) {
    if (index >= inputString.size() || !isUtf8ContinuationByte(static_cast<unsigned char>(inputString[index]))) {
        throw std::runtime_error("Invalid UTF-8");
    }
    return static_cast<unsigned char>(inputString[index]) & 0x3F;
}
#endif

std::wstring stringToWstring(const std::string &inputString) {
    if (inputString.empty()) {
        return {};
    }

#if defined(_WIN32)
    if (inputString.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("String is too large to convert to UTF-16");
    }

    const int len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, inputString.data(),
                                        static_cast<int>(inputString.size()), nullptr, 0);
    if (len == 0) {
        throw std::runtime_error("Invalid UTF-8");
    }

    std::wstring wide(static_cast<std::size_t>(len), 0);
    const int converted = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, inputString.data(),
                                              static_cast<int>(inputString.size()), wide.data(), len);
    if (converted != len) {
        throw std::runtime_error("Invalid UTF-8");
    }
    return wide;
#else
    std::wstring wide;
    wide.reserve(inputString.size());

    for (std::size_t i = 0; i < inputString.size();) {
        int codePoint = 0;
        const auto c = static_cast<unsigned char>(inputString[i]);

        if (c < 0x80) {
            codePoint = c;
            i += 1;
        } else if ((c & 0xE0U) == 0xC0U) {
            if (c < 0xC2U) {
                throw std::runtime_error("Invalid UTF-8");
            }
            codePoint = ((c & 0x1F) << 6) | utf8ContinuationPayload(inputString, i + 1);
            i += 2;
        } else if ((c & 0xF0U) == 0xE0U) {
            const auto second = utf8ContinuationPayload(inputString, i + 1);
            if ((c == 0xE0U && second < 0x20) || (c == 0xEDU && second >= 0x20)) {
                throw std::runtime_error("Invalid UTF-8");
            }
            codePoint = ((c & 0x0F) << 12) | (second << 6) | utf8ContinuationPayload(inputString, i + 2);
            i += 3;
        } else if ((c & 0xF8U) == 0xF0U) {
            if (c > 0xF4U) {
                throw std::runtime_error("Invalid UTF-8");
            }
            const auto second = utf8ContinuationPayload(inputString, i + 1);
            if ((c == 0xF0U && second < 0x10) || (c == 0xF4U && second >= 0x10)) {
                throw std::runtime_error("Invalid UTF-8");
            }
            codePoint = ((c & 0x07) << 18) | (second << 12) | (utf8ContinuationPayload(inputString, i + 2) << 6) |
                        utf8ContinuationPayload(inputString, i + 3);
            i += 4;
        } else {
            throw std::runtime_error("Invalid UTF-8");
        }

        wide.push_back(static_cast<wchar_t>(codePoint));
    }

    return wide;
#endif
}

std::vector<DxcDefine> parsePreprocessorOptions(const std::string &options, std::vector<std::wstring> &storage) {
    std::istringstream stream{options};
    std::string option;
    std::vector<DxcDefine> defines;
    while (stream >> option) {
        if (option.rfind("-D", 0) != 0 || option.size() <= 2) {
            throw std::runtime_error("Unsupported HLSL build option '" + option + "'");
        }

        const auto definition = option.substr(2);
        const auto equal = definition.find('=');
        DxcDefine define{};
        if (equal == std::string::npos) {
            storage.emplace_back(stringToWstring(definition));
            define.Name = storage.back().c_str();
        } else {
            const auto nameIndex = storage.size();
            storage.emplace_back(stringToWstring(definition.substr(0, equal)));
            const auto valueIndex = storage.size();
            storage.emplace_back(stringToWstring(definition.substr(equal + 1)));
            define.Name = storage[nameIndex].c_str();
            define.Value = storage[valueIndex].c_str();
        }
        defines.push_back(define);
    }
    return defines;
}

void ensureStaticDxcInitialized() {
    static std::once_flag initFlag;
    static HRESULT initResult = S_OK;

    std::call_once(initFlag, []() {
        bool fileSystemSetup = false;

        initResult = DxcInitThreadMalloc();
        if (FAILED(initResult)) {
            return;
        }

        DxcSetThreadMallocToDefault();
        if (::llvm::sys::fs::SetupPerThreadFileSystem()) {
            initResult = E_FAIL;
            goto cleanup;
        }
        fileSystemSetup = true;

        initResult = ::hlsl::SetupRegistryPassForHLSL();
        if (FAILED(initResult)) {
            goto cleanup;
        }

        initResult = ::hlsl::SetupRegistryPassForPIX();
        if (FAILED(initResult)) {
            goto cleanup;
        }

        if (::hlsl::options::initHlslOptTable()) {
            initResult = E_FAIL;
            goto cleanup;
        }

    cleanup:
        DxcClearThreadMalloc();
        if (FAILED(initResult)) {
            if (fileSystemSetup) {
                ::llvm::sys::fs::CleanupPerThreadFileSystem();
            }
            DxcCleanupThreadMalloc();
        }
    });

    if (FAILED(initResult)) {
        throw std::runtime_error("Failed to initialize statically linked DXC runtime");
    }
}

std::string dxcOutput(IDxcResult &result, DXC_OUT_KIND outputKind) {
    CComPtr<IDxcBlobUtf8> blob;
    const auto outputResult = result.GetOutput(outputKind, IID_PPV_ARGS(&blob), nullptr);
    if (FAILED(outputResult) || blob == nullptr || blob->GetStringLength() == 0) {
        return {};
    }
    return {blob->GetStringPointer(), blob->GetStringLength()};
}

std::vector<const wchar_t *> includeArguments(const std::vector<std::wstring> &includeDirs) {
    std::vector<const wchar_t *> args;
    args.reserve(includeDirs.size());
    for (const auto &includeDir : includeDirs) {
        args.push_back(includeDir.c_str());
    }
    return args;
}

} // namespace

std::vector<uint32_t> compileHlslComputeToSpirv(const Module &module) {
    auto result = HlslCompiler::get().compile(module.source, module.entryPoint, module.name, module.buildOptions,
                                              moduleIncludeDirectoryStrings(module));
    if (result.second.empty()) {
        throw std::runtime_error("HLSL module '" + module.name +
                                 "' compilation produced empty SPIR-V: " + result.first);
    }
    return std::move(result.second);
}

HlslCompiler &HlslCompiler::get() {
    static HlslCompiler hlslCompiler;
    return hlslCompiler;
}

std::pair<std::string, std::vector<uint32_t>>
HlslCompiler::compile(const std::string &source, const std::string &entryPoint, const std::string &debugName,
                      const std::string &preprocessorOptions, const std::vector<std::string> &shaderDirs) {
    ensureStaticDxcInitialized();

    CComPtr<IDxcUtils> utils;
    CComPtr<IDxcCompiler3> compiler;
    HRESULT result = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
    if (FAILED(result) || utils == nullptr) {
        throw std::runtime_error("Failed to create IDxcUtils instance");
    }
    result = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));
    if (FAILED(result) || compiler == nullptr) {
        throw std::runtime_error("Failed to create IDxcCompiler3 instance");
    }

    DxcBuffer buffer{};
    buffer.Ptr = source.data();
    buffer.Size = source.size();
    buffer.Encoding = DXC_CP_UTF8;

    CComPtr<IDxcIncludeHandler> includeHandler;
    result = utils->CreateDefaultIncludeHandler(&includeHandler);
    if (FAILED(result) || includeHandler == nullptr) {
        throw std::runtime_error("Failed to create HLSL include handler");
    }

    const auto name = stringToWstring(debugName);
    const auto entry = stringToWstring(entryPoint);

    std::vector<std::wstring> includeDirStorage;
    includeDirStorage.reserve(shaderDirs.size());
    for (const auto &dir : shaderDirs) {
        includeDirStorage.emplace_back(stringToWstring(dir));
    }
    auto compileArgs = includeArguments(includeDirStorage);
    compileArgs.push_back(L"-spirv");
    compileArgs.push_back(L"-enable-16bit-types");
    compileArgs.push_back(L"-Wno-conversion");

    std::vector<std::wstring> defineStorage;
    auto defines = parsePreprocessorOptions(preprocessorOptions, defineStorage);

    CComPtr<IDxcCompilerArgs> args;
    result = utils->BuildArguments(
        name.c_str(), entry.c_str(), L"cs_6_2", compileArgs.data(), static_cast<uint32_t>(compileArgs.size()),
        defines.empty() ? nullptr : defines.data(), static_cast<uint32_t>(defines.size()), &args);
    if (FAILED(result) || args == nullptr) {
        throw std::runtime_error("Failed to build HLSL compiler arguments");
    }

    CComPtr<IDxcResult> compileResult;
    result = compiler->Compile(&buffer, args->GetArguments(), args->GetCount(), includeHandler,
                               IID_PPV_ARGS(&compileResult));
    if (FAILED(result) || compileResult == nullptr) {
        throw std::runtime_error("Failed to invoke HLSL compiler");
    }

    const auto log = dxcOutput(*compileResult, DXC_OUT_ERRORS);
    HRESULT compileStatus = S_OK;
    result = compileResult->GetStatus(&compileStatus);
    if (FAILED(result)) {
        throw std::runtime_error("Failed to query HLSL compilation status");
    }
    if (FAILED(compileStatus)) {
        return {log.empty() ? "HLSL compilation failed" : log, {}};
    }

    CComPtr<IDxcBlob> object;
    result = compileResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object), nullptr);
    if (FAILED(result) || object == nullptr) {
        return {"HLSL compilation produced no object output", {}};
    }

    const auto *data = object->GetBufferPointer();
    const auto size = object->GetBufferSize();
    if (size % sizeof(uint32_t) != 0) {
        return {"HLSL object blob size is not a multiple of 4 bytes", {}};
    }

    std::vector<uint32_t> spirv(size / sizeof(uint32_t));
    std::memcpy(spirv.data(), data, size);
    return {log, spirv};
}

namespace {

// Namespace-scope initialization registers this backend when its object file is
// linked.
const bool registeredHlslCompiler = SourceModuleCompilerRegistration(ModuleCodeKind::Hlsl, compileHlslComputeToSpirv);

} // namespace

} // namespace mlsdk::workloadlib::detail
