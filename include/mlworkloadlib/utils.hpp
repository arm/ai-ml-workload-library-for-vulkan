/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>

namespace mlworkloadlib {

template <typename T> class ArrayView {
  public:
    using pointer = T *;
    using reference = T &;
    using iterator = T *;

    constexpr ArrayView() noexcept = default;
    constexpr ArrayView(pointer data, std::size_t size) noexcept : data_(data), size_(size) {}

    constexpr pointer data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    constexpr iterator begin() const noexcept { return data_; }
    constexpr iterator end() const noexcept { return data_ == nullptr ? nullptr : data_ + size_; }
    constexpr reference operator[](std::size_t index) const noexcept { return data_[index]; }

  private:
    pointer data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace mlworkloadlib
