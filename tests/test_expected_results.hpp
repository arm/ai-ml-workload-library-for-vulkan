/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "test_vulkan_resources.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace mlworkloadlib::test {

/*******************************************************************************
 * Expected result helpers
 *******************************************************************************/

inline std::vector<int32_t> addVectors(const std::vector<int32_t> &lhs, const std::vector<int32_t> &rhs) {
    std::vector<int32_t> result(lhs.size());
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), std::plus<>());
    return result;
}

inline std::vector<int32_t> int32WordsFromBytes(const std::vector<int8_t> &bytes, std::size_t words) {
    std::vector<int32_t> result(words);
    for (std::size_t word = 0; word < words; ++word) {
        uint32_t value = 0;
        for (std::size_t byte = 0; byte < sizeof(int32_t); ++byte) {
            value |= static_cast<uint32_t>(static_cast<uint8_t>(bytes[(word * sizeof(int32_t)) + byte])) << (byte * 8);
        }
        result[word] = static_cast<int32_t>(value);
    }
    return result;
}

inline std::size_t nhwcIndex(int64_t batchIndex, int64_t heightIndex, int64_t widthIndex, int64_t channelIndex,
                             int64_t heightExtent, int64_t widthExtent, int64_t channelExtent) {
    return static_cast<std::size_t>(
        (((((batchIndex * heightExtent) + heightIndex) * widthExtent) + widthIndex) * channelExtent) + channelIndex);
}

inline std::vector<int8_t> makeMaxpoolInput(const std::vector<int64_t> &shape, uint32_t seed = 0) {
    const auto batch = shape[0];
    const auto height = shape[1];
    const auto width = shape[2];
    const auto channels = shape[3];

    std::vector<int8_t> input(Tensor::numElements(shape));
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t h = 0; h < height; ++h) {
            for (int64_t w = 0; w < width; ++w) {
                for (int64_t c = 0; c < channels; ++c) {
                    const auto index = nhwcIndex(n, h, w, c, height, width, channels);
                    input[index] = static_cast<int8_t>((n * 17 + h * 13 + w * 7 + c * 3 + seed) % 97);
                }
            }
        }
    }
    return input;
}

inline std::vector<int8_t> expectedMaxpool(const std::vector<int8_t> &input, const std::vector<int64_t> &shape) {
    const auto batch = shape[0];
    const auto inputHeight = shape[1];
    const auto inputWidth = shape[2];
    const auto channels = shape[3];
    const auto outputHeight = inputHeight / 2;
    const auto outputWidth = inputWidth / 2;

    std::vector<int8_t> expected(Tensor::numElements({batch, outputHeight, outputWidth, channels}));
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t h = 0; h < outputHeight; ++h) {
            for (int64_t w = 0; w < outputWidth; ++w) {
                for (int64_t c = 0; c < channels; ++c) {
                    const auto firstInputIndex = nhwcIndex(n, h * 2, w * 2, c, inputHeight, inputWidth, channels);
                    int8_t maxValue = input[firstInputIndex];
                    for (int64_t kh = 0; kh < 2; ++kh) {
                        for (int64_t kw = 0; kw < 2; ++kw) {
                            const auto inputIndex =
                                nhwcIndex(n, (h * 2) + kh, (w * 2) + kw, c, inputHeight, inputWidth, channels);
                            maxValue = std::max(maxValue, input[inputIndex]);
                        }
                    }
                    expected[nhwcIndex(n, h, w, c, outputHeight, outputWidth, channels)] = maxValue;
                }
            }
        }
    }
    return expected;
}

} // namespace mlworkloadlib::test
