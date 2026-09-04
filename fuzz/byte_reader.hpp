/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace mlworkloadlib::fuzz {

class ByteReader {
  public:
    ByteReader(const uint8_t *data, std::size_t size) : data_(data), size_(size) {}

    uint8_t consumeByte(uint8_t fallback = 0) {
        if (offset_ >= size_) {
            return fallback;
        }
        return data_[offset_++];
    }

    uint32_t consumeU32() {
        uint32_t value = 0;
        for (uint32_t byteIndex = 0; byteIndex < 4; ++byteIndex) {
            value = static_cast<uint32_t>((value << 8U) | static_cast<uint32_t>(consumeByte()));
        }
        return value;
    }

    uint32_t choose(uint32_t choiceCount) {
        if (choiceCount == 0) {
            return 0;
        }
        return consumeU32() % choiceCount;
    }

    bool boolean() { return (consumeByte() & 1U) != 0; }

  private:
    const uint8_t *data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t offset_ = 0;
};

} // namespace mlworkloadlib::fuzz
