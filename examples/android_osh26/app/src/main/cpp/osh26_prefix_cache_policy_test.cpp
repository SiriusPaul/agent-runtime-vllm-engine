#include "osh26_prefix_cache_policy.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<int> seq(int count, int base = 1000) {
    std::vector<int> out;
    out.reserve((size_t) count);
    for (int i = 0; i < count; ++i) {
        out.push_back(base + i);
    }
    return out;
}

bool expect_reuse(
        const char * name,
        const std::vector<int> & prompt,
        const std::vector<int> & cached,
        size_t cached_pages,
        size_t expected) {
    constexpr size_t page_tokens = 16;
    const size_t actual = osh26::reusable_paged_prefix_tokens(prompt, cached, cached_pages, page_tokens);
    if (actual == expected) {
        return true;
    }
    std::cerr << name << ": expected " << expected << " reusable tokens, got " << actual << "\n";
    return false;
}

} // namespace

int main() {
    bool ok = true;

    ok &= expect_reuse("empty cache", seq(33), {}, 0, 0);
    ok &= expect_reuse("33 token exact match keeps final prompt token fresh", seq(33), seq(33), 2, 32);
    ok &= expect_reuse("32 token exact match recomputes tail page", seq(32), seq(32), 2, 16);

    auto diff_at_20 = seq(40);
    diff_at_20[20] += 10000;
    ok &= expect_reuse("mismatch after one full page", diff_at_20, seq(40), 2, 16);

    auto diff_at_8 = seq(40);
    diff_at_8[8] += 10000;
    ok &= expect_reuse("mismatch before first page", diff_at_8, seq(40), 2, 0);

    auto tail_diff = seq(48);
    tail_diff[40] += 10000;
    ok &= expect_reuse("tail mismatch reuses full common pages", tail_diff, seq(48), 3, 32);

    ok &= expect_reuse("short prompt below one fresh-tail page", seq(16), seq(16), 1, 0);
    ok &= expect_reuse("cached entry longer than prompt", seq(41), seq(80), 5, 32);
    ok &= expect_reuse("cached pages cap reusable tokens", seq(80), seq(80), 3, 48);
    ok &= expect_reuse("strict token equality rejects same length different first token", seq(40, 2000), seq(40, 3000), 2, 0);

    if (!ok) {
        return EXIT_FAILURE;
    }
    std::cout << "prefix cache policy token cases passed\n";
    return EXIT_SUCCESS;
}
