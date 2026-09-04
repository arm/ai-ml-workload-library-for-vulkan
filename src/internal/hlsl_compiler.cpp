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
#include <mutex>
#include <new>
#include <sstream>

#if defined(_WIN32)
#    include <atlbase.h>
#    include <windows.h>
#endif

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hlsl {
HRESULT SetupRegistryPassForHLSL();
HRESULT SetupRegistryPassForPIX();
} // namespace hlsl

#if defined(_WIN32) && !defined(DXC_DISABLE_ALLOCATOR_OVERRIDES)
void *__CRTDECL operator new(std::size_t size) noexcept(false) {
    void *ptr = DxcNew(size);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void *__CRTDECL operator new(std::size_t size, const std::nothrow_t &) throw() { return DxcNew(size); }

void __CRTDECL operator delete(void *ptr) throw() { DxcDelete(ptr); }

void __CRTDECL operator delete(void *ptr, const std::nothrow_t &) throw() { DxcDelete(ptr); }
#endif

namespace mlworkloadlib::detail {
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

std::wstring stringToWstring(const std::string &inputString) {
#if defined(_WIN32)
    const int len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, inputString.data(),
                                        static_cast<int>(inputString.size()), nullptr, 0);
    if (len < 0) {
        throw std::runtime_error("Invalid UTF-8");
    }

    std::wstring wide(len, 0);
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, inputString.data(), static_cast<int>(inputString.size()),
                        wide.data(), len);
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
        } else if ((c >> 5) == 0x6) {
            codePoint = ((c & 0x1F) << 6) | (inputString[i + 1] & 0x3F);
            i += 2;
        } else if ((c >> 4) == 0xE) {
            codePoint = ((c & 0x0F) << 12) | ((inputString[i + 1] & 0x3F) << 6) | (inputString[i + 2] & 0x3F);
            i += 3;
        } else if ((c >> 3) == 0x1E) {
            codePoint = ((c & 0x07) << 18) | ((inputString[i + 1] & 0x3F) << 12) | ((inputString[i + 2] & 0x3F) << 6) |
                        (inputString[i + 3] & 0x3F);
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
    result.GetOutput(outputKind, IID_PPV_ARGS(&blob), nullptr);
    if (blob == nullptr || blob->GetStringLength() == 0) {
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

std::vector<std::string> includeDirStrings(const std::vector<std::filesystem::path> &includeDirs) {
    std::vector<std::string> result;
    result.reserve(includeDirs.size());
    for (const auto &includeDir : includeDirs) {
        result.push_back(includeDir.string());
    }
    return result;
}

} // namespace

std::vector<uint32_t> compileHlslComputeToSpirv(const Module &module) {
    auto result = HlslCompiler::get().compile(module.source, module.entryPoint, module.name, module.buildOptions,
                                              includeDirStrings(module.includeDirs));
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
    utils->CreateDefaultIncludeHandler(&includeHandler);

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
    utils->BuildArguments(name.c_str(), entry.c_str(), L"cs_6_2", compileArgs.data(),
                          static_cast<uint32_t>(compileArgs.size()), defines.data(),
                          static_cast<uint32_t>(defines.size()), &args);

    CComPtr<IDxcResult> compileResult;
    compiler->Compile(&buffer, args->GetArguments(), args->GetCount(), includeHandler, IID_PPV_ARGS(&compileResult));
    const auto log = dxcOutput(*compileResult, DXC_OUT_ERRORS);
    if (!log.empty()) {
        return {log, {}};
    }

    CComPtr<IDxcBlob> object;
    compileResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object), nullptr);
    if (object == nullptr) {
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

} // namespace mlworkloadlib::detail
