/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>

namespace mlsdk::workloadlib {

/**
 * @brief Non-owning view over a contiguous sequence.
 * @tparam T Element type, which may be const-qualified.
 */
template <typename T> class ArrayView {
  public:
    using pointer = T *;
    using reference = T &;
    using iterator = T *;

    /** @brief Constructs an empty view. */
    constexpr ArrayView() noexcept = default;

    /** @brief Constructs a view from a pointer and element count. */
    constexpr ArrayView(pointer data, std::size_t size) noexcept : data_(data), size_(size) {}

    /** @brief Returns the pointer to the first element. */
    constexpr pointer data() const noexcept { return data_; }

    /** @brief Returns the number of elements. */
    constexpr std::size_t size() const noexcept { return size_; }

    /** @brief Returns true when size() is zero. */
    constexpr bool empty() const noexcept { return size_ == 0; }

    /** @brief Returns an iterator to the first element. */
    constexpr iterator begin() const noexcept { return data_; }

    /** @brief Returns the past-the-end iterator. */
    constexpr iterator end() const noexcept { return data_ == nullptr ? nullptr : data_ + size_; }

    /** @brief Returns an element without bounds checking. */
    constexpr reference operator[](std::size_t index) const noexcept { return data_[index]; }

  private:
    pointer data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace mlsdk::workloadlib
