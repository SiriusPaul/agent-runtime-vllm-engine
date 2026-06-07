#pragma once

#include <algorithm>
#include <cstddef>

namespace osh26 {

template <typename TokensA, typename TokensB>
constexpr size_t strict_common_prefix_tokens(const TokensA & lhs, const TokensB & rhs) {
    const size_t limit = std::min(lhs.size(), rhs.size());
    size_t n = 0;
    while (n < limit && lhs[n] == rhs[n]) {
        ++n;
    }
    return n;
}

template <typename PromptTokens, typename CachedTokens>
constexpr size_t reusable_paged_prefix_tokens(
        const PromptTokens & prompt_tokens,
        const CachedTokens & cached_tokens,
        size_t cached_page_count,
        size_t page_tokens) {
    if (page_tokens == 0 || cached_page_count == 0 || prompt_tokens.size() <= 1 || cached_tokens.empty()) {
        return 0;
    }

    const size_t common = strict_common_prefix_tokens(prompt_tokens, cached_tokens);
    const size_t prompt_cap = prompt_tokens.size() - 1;
    const size_t cached_cap = cached_page_count * page_tokens;
    const size_t capped = std::min(std::min(common, prompt_cap), cached_cap);
    return (capped / page_tokens) * page_tokens;
}

} // namespace osh26
