/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "glsl_compiler.hpp"
#include "module_compiler.hpp"
#include "workload_impl.hpp"

#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/DirStackFileIncluder.h>
#include <glslang/Public/ShaderLang.h>

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern const TBuiltInResource *GetDefaultResources();

namespace mlworkloadlib::detail {
namespace {

class GlslangProcess {
  public:
    GlslangProcess(const GlslangProcess &) = delete;
    GlslangProcess &operator=(const GlslangProcess &) = delete;
    GlslangProcess(GlslangProcess &&) = delete;
    GlslangProcess &operator=(GlslangProcess &&) = delete;

    static void ensureInitialized() {
        static const GlslangProcess process;
        (void)process;
    }

  private:
    GlslangProcess() { glslang::InitializeProcess(); }
    ~GlslangProcess() { glslang::FinalizeProcess(); }
};

std::string parsePreprocessorOptions(std::string_view options) {
    std::istringstream stream{std::string(options)};
    std::string option;
    std::string preamble;
    while (stream >> option) {
        if (option.rfind("-D", 0) != 0 || option.size() <= 2) {
            throw std::runtime_error("Unsupported GLSL build option '" + option + "'");
        }

        auto define = option.substr(2);
        const auto equal = define.find('=');
        if (equal != std::string::npos) {
            define[equal] = ' ';
        }
        preamble += "#define " + define + "\n";
    }
    return preamble;
}

std::vector<std::string> includeDirStrings(const std::vector<std::filesystem::path> &includeDirs) {
    std::vector<std::string> result;
    result.reserve(includeDirs.size());
    for (const auto &includeDir : includeDirs) {
        result.push_back(includeDir.string());
    }
    return result;
}

std::string compilerLog(glslang::TShader &shader) {
    return std::string(shader.getInfoLog()) + "\n" + std::string(shader.getInfoDebugLog());
}

} // namespace

std::vector<uint32_t> compileGlslComputeToSpirv(const Module &module) {
    GlslangProcess::ensureInitialized();

    constexpr auto language = EShLanguage::EShLangCompute;
    const auto messages = static_cast<EShMessages>(EShMsgDefault | EShMsgVulkanRules | EShMsgSpvRules);

    glslang::TShader shader(language);
    shader.setEnvInput(glslang::EShSourceGlsl, language, glslang::EShClientVulkan, 110);
    shader.setEnvClient(glslang::EShClient::EShClientVulkan, glslang::EShTargetVulkan_1_3);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);

    const char *source = module.source.c_str();
    shader.setStrings(&source, 1);

    const auto includeDirs = includeDirStrings(module.includeDirs);
    DirStackFileIncluder includer;
    for (const auto &includeDir : includeDirs) {
        includer.pushExternalDirectory(includeDir);
    }

    const auto preamble = parsePreprocessorOptions(module.buildOptions);
    shader.setPreamble(preamble.c_str());

    std::string preprocessed;
    if (!shader.preprocess(GetDefaultResources(), 460, ENoProfile, false, false, messages, &preprocessed, includer)) {
        throw std::runtime_error("GLSL module preprocessing failed: " + compilerLog(shader));
    }

    if (!shader.parse(GetDefaultResources(), 460, false, messages, includer)) {
        throw std::runtime_error("GLSL module compilation failed: " + compilerLog(shader));
    }

    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages)) {
        throw std::runtime_error("GLSL module linking failed: " + std::string(program.getInfoLog()) + "\n" +
                                 std::string(program.getInfoDebugLog()));
    }

    const auto *intermediate = program.getIntermediate(language);
    if (intermediate == nullptr) {
        throw std::runtime_error("GLSL module compilation produced no intermediate representation");
    }

    glslang::SpvOptions spvOptions;
    spv::SpvBuildLogger spvLogger;
    std::vector<uint32_t> spirv;
    glslang::GlslangToSpv(*intermediate, spirv, &spvLogger, &spvOptions);
    if (spirv.empty()) {
        throw std::runtime_error("GLSL module compilation produced empty SPIR-V: " + spvLogger.getAllMessages());
    }
    return spirv;
}

namespace {

// Namespace-scope initialization registers this backend when its object file is
// linked.
const bool registeredGlslCompiler = SourceModuleCompilerRegistration(ModuleCodeKind::Glsl, compileGlslComputeToSpirv);

} // namespace

} // namespace mlworkloadlib::detail
