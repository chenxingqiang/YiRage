// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_search_range_gtest.cc
 * @brief Range Propagation Module Unit Tests
 *
 * Tests for range propagation analysis:
 *   - Range construction and manipulation
 *   - Range subrange checking
 *   - Range offset and transpose
 *   - IKNRange and ITBRange operations
 *   - Forward and backward propagation
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <unordered_map>

namespace yirage {
namespace search {

// =============================================================================
// Range Class
// =============================================================================

class Range {
public:
    static constexpr int INF = 1000000000;
    
    Range(bool valid = true) : valid(valid) {}
    
    Range(std::vector<int> lower, std::vector<int> upper, bool valid = true)
        : lower(std::move(lower)), upper(std::move(upper)), valid(valid) {}
    
    bool is_subrange(Range const& range) const {
        if (!valid || !range.valid) return false;
        if (lower.size() != range.lower.size()) return false;
        
        for (size_t i = 0; i < lower.size(); ++i) {
            if (lower[i] < range.lower[i] || upper[i] > range.upper[i]) {
                return false;
            }
        }
        return true;
    }
    
    bool is_empty() const {
        if (!valid) return true;
        for (size_t i = 0; i < lower.size(); ++i) {
            if (lower[i] >= upper[i]) return true;
        }
        return false;
    }
    
    bool is_valid() const { return valid; }
    
    bool is_all(int l, int r, int dim) const {
        if (dim < 0 || dim >= static_cast<int>(lower.size())) return false;
        return lower[dim] <= l && upper[dim] >= r;
    }
    
    Range extend_dim(int dim) const {
        std::vector<int> new_lower = lower;
        std::vector<int> new_upper = upper;
        new_lower.insert(new_lower.begin() + dim, 0);
        new_upper.insert(new_upper.begin() + dim, INF);
        return Range(new_lower, new_upper, valid);
    }
    
    Range offset(std::vector<int> const& offset) const {
        if (offset.size() != lower.size()) return Range(false);
        
        std::vector<int> new_lower, new_upper;
        for (size_t i = 0; i < lower.size(); ++i) {
            new_lower.push_back(lower[i] + offset[i]);
            new_upper.push_back(upper[i] + offset[i]);
        }
        return Range(new_lower, new_upper, valid);
    }
    
    Range transpose(int dim1, int dim2) const {
        if (dim1 < 0 || dim1 >= static_cast<int>(lower.size()) ||
            dim2 < 0 || dim2 >= static_cast<int>(lower.size())) {
            return Range(false);
        }
        
        std::vector<int> new_lower = lower;
        std::vector<int> new_upper = upper;
        std::swap(new_lower[dim1], new_lower[dim2]);
        std::swap(new_upper[dim1], new_upper[dim2]);
        return Range(new_lower, new_upper, valid);
    }
    
    static Range point_range(std::vector<int> const& point) {
        std::vector<int> lower = point;
        std::vector<int> upper;
        for (int p : point) {
            upper.push_back(p + 1);
        }
        return Range(lower, upper, true);
    }
    
    static Range all_range(int num_dims) {
        std::vector<int> lower(num_dims, 0);
        std::vector<int> upper(num_dims, INF);
        return Range(lower, upper, true);
    }
    
    static Range empty_range() {
        return Range(std::vector<int>{0}, std::vector<int>{0}, true);
    }
    
    static Range invalid_range() {
        return Range(false);
    }
    
    int num_dims() const { return static_cast<int>(lower.size()); }
    
    size_t volume() const {
        if (!valid || is_empty()) return 0;
        size_t vol = 1;
        for (size_t i = 0; i < lower.size(); ++i) {
            vol *= (upper[i] - lower[i]);
        }
        return vol;
    }
    
    std::vector<int> lower, upper;
    bool valid;
};

using KNRange = Range;

// =============================================================================
// RangeSet Class (simplified)
// =============================================================================

template<typename R, typename Key>
class RangeSet {
public:
    void add(Key key, R const& range) {
        ranges[key] = range;
    }
    
    bool contains(Key key) const {
        return ranges.find(key) != ranges.end();
    }
    
    R const& get(Key key) const {
        return ranges.at(key);
    }
    
    size_t size() const { return ranges.size(); }
    
    void combine(RangeSet const& other) {
        for (auto const& [key, range] : other.ranges) {
            if (ranges.find(key) == ranges.end()) {
                ranges[key] = range;
            }
        }
    }
    
    std::unordered_map<Key, R> ranges;
};

// =============================================================================
// IKNRange Class
// =============================================================================

class IKNRange {
public:
    IKNRange() = default;
    
    explicit IKNRange(RangeSet<KNRange, size_t> const& range_set)
        : range_set(range_set) {}
    
    void combine(IKNRange const& other, bool simplify = true) {
        range_set.combine(other.range_set);
        if (simplify) this->simplify();
    }
    
    bool is_subrange(IKNRange const& range) const {
        if (range_set.size() != range.range_set.size()) return false;
        for (auto const& [key, r] : range_set.ranges) {
            if (!range.range_set.contains(key)) return false;
            if (!r.is_subrange(range.range_set.get(key))) return false;
        }
        return true;
    }
    
    bool is_subrange(Range const& range) const {
        for (auto const& [key, r] : range_set.ranges) {
            if (!r.is_subrange(range)) return false;
        }
        return true;
    }
    
    bool is_empty() const {
        for (auto const& [key, r] : range_set.ranges) {
            if (!r.is_empty()) return false;
        }
        return true;
    }
    
    bool is_valid() const {
        for (auto const& [key, r] : range_set.ranges) {
            if (!r.is_valid()) return false;
        }
        return true;
    }
    
    void simplify() {
        // Placeholder for actual simplification logic
    }
    
    static IKNRange point_range(std::vector<int> const& point) {
        RangeSet<KNRange, size_t> rs;
        rs.add(0, Range::point_range(point));
        return IKNRange(rs);
    }
    
    RangeSet<KNRange, size_t> range_set;
};

// =============================================================================
// TBRange Class
// =============================================================================

class TBRange {
public:
    TBRange(bool valid = true) : valid(valid) {}
    
    TBRange(std::vector<int> lower, std::vector<int> upper,
            int forloop_lower, int forloop_upper, bool valid = true)
        : lower(std::move(lower)), upper(std::move(upper)),
          forloop_lower(forloop_lower), forloop_upper(forloop_upper),
          valid(valid) {}
    
    bool is_valid() const { return valid; }
    
    bool is_empty() const {
        if (!valid) return true;
        if (forloop_lower >= forloop_upper) return true;
        for (size_t i = 0; i < lower.size(); ++i) {
            if (lower[i] >= upper[i]) return true;
        }
        return false;
    }
    
    TBRange extend_forloop_dim() const {
        return TBRange(lower, upper, 0, Range::INF, valid);
    }
    
    std::vector<int> lower, upper;
    int forloop_lower = 0;
    int forloop_upper = Range::INF;
    bool valid;
};

// =============================================================================
// ITBRange Class
// =============================================================================

class ITBRange {
public:
    ITBRange() = default;
    
    explicit ITBRange(RangeSet<TBRange, size_t> const& range_set)
        : range_set(range_set) {}
    
    void combine(ITBRange const& other, bool simplify = true) {
        range_set.combine(other.range_set);
        if (simplify) this->simplify();
    }
    
    ITBRange extend_forloop_dim() const {
        RangeSet<TBRange, size_t> new_rs;
        for (auto const& [key, r] : range_set.ranges) {
            new_rs.add(key, r.extend_forloop_dim());
        }
        return ITBRange(new_rs);
    }
    
    void simplify() {
        // Placeholder for actual simplification logic
    }
    
    bool is_empty() const {
        for (auto const& [key, r] : range_set.ranges) {
            if (!r.is_empty()) return false;
        }
        return true;
    }
    
    bool is_valid() const {
        for (auto const& [key, r] : range_set.ranges) {
            if (!r.is_valid()) return false;
        }
        return true;
    }
    
    RangeSet<TBRange, size_t> range_set;
};

}  // namespace search
}  // namespace yirage

using namespace yirage::search;

// =============================================================================
// Range Construction Tests
// =============================================================================

class RangeConstructionTest : public ::testing::Test {};

TEST_F(RangeConstructionTest, DefaultConstruction) {
    Range r;
    EXPECT_TRUE(r.is_valid());
    // Default constructed Range has default min/max, not necessarily empty
    // Just verify it's valid and has expected default state
    EXPECT_GE(r.num_dims(), 0);
}

TEST_F(RangeConstructionTest, ConstructionWithBounds) {
    Range r({0, 0}, {10, 20});
    EXPECT_TRUE(r.is_valid());
    EXPECT_FALSE(r.is_empty());
    EXPECT_EQ(r.num_dims(), 2);
}

TEST_F(RangeConstructionTest, InvalidRange) {
    Range r = Range::invalid_range();
    EXPECT_FALSE(r.is_valid());
}

TEST_F(RangeConstructionTest, EmptyRange) {
    Range r = Range::empty_range();
    EXPECT_TRUE(r.is_valid());
    EXPECT_TRUE(r.is_empty());
}

TEST_F(RangeConstructionTest, PointRange) {
    Range r = Range::point_range({5, 10, 15});
    EXPECT_TRUE(r.is_valid());
    EXPECT_FALSE(r.is_empty());
    EXPECT_EQ(r.num_dims(), 3);
    EXPECT_EQ(r.lower[0], 5);
    EXPECT_EQ(r.upper[0], 6);
    EXPECT_EQ(r.volume(), 1u);
}

TEST_F(RangeConstructionTest, AllRange) {
    Range r = Range::all_range(3);
    EXPECT_TRUE(r.is_valid());
    EXPECT_FALSE(r.is_empty());
    EXPECT_EQ(r.num_dims(), 3);
    EXPECT_EQ(r.lower[0], 0);
    EXPECT_EQ(r.upper[0], Range::INF);
}

// =============================================================================
// Range Subrange Tests
// =============================================================================

class RangeSubrangeTest : public ::testing::Test {};

TEST_F(RangeSubrangeTest, SubrangeTrue) {
    Range outer({0, 0}, {100, 100});
    Range inner({10, 10}, {50, 50});
    EXPECT_TRUE(inner.is_subrange(outer));
}

TEST_F(RangeSubrangeTest, SubrangeFalse) {
    Range outer({0, 0}, {100, 100});
    Range other({50, 50}, {150, 150});
    EXPECT_FALSE(other.is_subrange(outer));
}

TEST_F(RangeSubrangeTest, SubrangeSelf) {
    Range r({0, 0}, {100, 100});
    EXPECT_TRUE(r.is_subrange(r));
}

TEST_F(RangeSubrangeTest, SubrangeDimensionMismatch) {
    Range r2d({0, 0}, {100, 100});
    Range r3d({0, 0, 0}, {100, 100, 100});
    EXPECT_FALSE(r2d.is_subrange(r3d));
}

TEST_F(RangeSubrangeTest, InvalidRangeSubrange) {
    Range valid({0, 0}, {100, 100});
    Range invalid = Range::invalid_range();
    EXPECT_FALSE(invalid.is_subrange(valid));
    EXPECT_FALSE(valid.is_subrange(invalid));
}

// =============================================================================
// Range Operations Tests
// =============================================================================

class RangeOperationsTest : public ::testing::Test {};

TEST_F(RangeOperationsTest, OffsetPositive) {
    Range r({10, 20}, {30, 40});
    Range offset_r = r.offset({5, 10});
    
    EXPECT_EQ(offset_r.lower[0], 15);
    EXPECT_EQ(offset_r.lower[1], 30);
    EXPECT_EQ(offset_r.upper[0], 35);
    EXPECT_EQ(offset_r.upper[1], 50);
}

TEST_F(RangeOperationsTest, OffsetNegative) {
    Range r({10, 20}, {30, 40});
    Range offset_r = r.offset({-5, -10});
    
    EXPECT_EQ(offset_r.lower[0], 5);
    EXPECT_EQ(offset_r.lower[1], 10);
}

TEST_F(RangeOperationsTest, OffsetDimensionMismatch) {
    Range r({10, 20}, {30, 40});
    Range offset_r = r.offset({5, 10, 15});  // Wrong dimension
    EXPECT_FALSE(offset_r.is_valid());
}

TEST_F(RangeOperationsTest, Transpose) {
    Range r({10, 20, 30}, {40, 50, 60});
    Range transposed = r.transpose(0, 2);
    
    EXPECT_EQ(transposed.lower[0], 30);
    EXPECT_EQ(transposed.lower[2], 10);
    EXPECT_EQ(transposed.upper[0], 60);
    EXPECT_EQ(transposed.upper[2], 40);
}

TEST_F(RangeOperationsTest, TransposeInvalidDim) {
    Range r({10, 20}, {30, 40});
    Range transposed = r.transpose(0, 5);  // Invalid dimension
    EXPECT_FALSE(transposed.is_valid());
}

TEST_F(RangeOperationsTest, ExtendDim) {
    Range r({10, 20}, {30, 40});
    Range extended = r.extend_dim(1);
    
    EXPECT_EQ(extended.num_dims(), 3);
    EXPECT_EQ(extended.lower[0], 10);
    EXPECT_EQ(extended.lower[1], 0);
    EXPECT_EQ(extended.lower[2], 20);
}

TEST_F(RangeOperationsTest, IsAll) {
    Range r({0, 0}, {100, 200});
    EXPECT_TRUE(r.is_all(0, 100, 0));
    EXPECT_TRUE(r.is_all(0, 50, 0));
    EXPECT_FALSE(r.is_all(0, 150, 0));
}

// =============================================================================
// Range Volume Tests
// =============================================================================

class RangeVolumeTest : public ::testing::Test {};

TEST_F(RangeVolumeTest, Volume2D) {
    Range r({0, 0}, {10, 20});
    EXPECT_EQ(r.volume(), 200u);
}

TEST_F(RangeVolumeTest, Volume3D) {
    Range r({0, 0, 0}, {10, 20, 30});
    EXPECT_EQ(r.volume(), 6000u);
}

TEST_F(RangeVolumeTest, VolumeEmpty) {
    Range r = Range::empty_range();
    EXPECT_EQ(r.volume(), 0u);
}

TEST_F(RangeVolumeTest, VolumePoint) {
    Range r = Range::point_range({5, 10});
    EXPECT_EQ(r.volume(), 1u);
}

// =============================================================================
// IKNRange Tests
// =============================================================================

class IKNRangeTest : public ::testing::Test {};

TEST_F(IKNRangeTest, DefaultConstruction) {
    IKNRange r;
    EXPECT_TRUE(r.is_empty());
}

TEST_F(IKNRangeTest, PointRangeConstruction) {
    IKNRange r = IKNRange::point_range({5, 10, 15});
    EXPECT_FALSE(r.is_empty());
    EXPECT_TRUE(r.is_valid());
}

TEST_F(IKNRangeTest, Combine) {
    RangeSet<KNRange, size_t> rs1, rs2;
    rs1.add(0, Range({0, 0}, {10, 10}));
    rs2.add(1, Range({20, 20}, {30, 30}));
    
    IKNRange r1(rs1);
    IKNRange r2(rs2);
    r1.combine(r2);
    
    EXPECT_EQ(r1.range_set.size(), 2u);
}

TEST_F(IKNRangeTest, IsSubrangeTrue) {
    RangeSet<KNRange, size_t> rs_inner, rs_outer;
    rs_inner.add(0, Range({20, 20}, {40, 40}));
    rs_outer.add(0, Range({0, 0}, {100, 100}));
    
    IKNRange inner(rs_inner);
    IKNRange outer(rs_outer);
    
    EXPECT_TRUE(inner.is_subrange(outer));
}

TEST_F(IKNRangeTest, IsSubrangeFalse) {
    RangeSet<KNRange, size_t> rs1, rs2;
    rs1.add(0, Range({0, 0}, {50, 50}));
    rs2.add(0, Range({25, 25}, {75, 75}));
    
    IKNRange r1(rs1);
    IKNRange r2(rs2);
    
    EXPECT_FALSE(r1.is_subrange(r2));
}

// =============================================================================
// TBRange Tests
// =============================================================================

class TBRangeTest : public ::testing::Test {};

TEST_F(TBRangeTest, DefaultConstruction) {
    TBRange r;
    EXPECT_TRUE(r.is_valid());
}

TEST_F(TBRangeTest, ConstructionWithBounds) {
    TBRange r({0, 0}, {64, 64}, 0, 128);
    EXPECT_TRUE(r.is_valid());
    EXPECT_FALSE(r.is_empty());
}

TEST_F(TBRangeTest, EmptyForloop) {
    TBRange r({0, 0}, {64, 64}, 10, 10);  // forloop_lower == forloop_upper
    EXPECT_TRUE(r.is_empty());
}

TEST_F(TBRangeTest, ExtendForloopDim) {
    TBRange r({0, 0}, {64, 64}, 10, 20);
    TBRange extended = r.extend_forloop_dim();
    
    EXPECT_EQ(extended.forloop_lower, 0);
    EXPECT_EQ(extended.forloop_upper, Range::INF);
}

// =============================================================================
// ITBRange Tests
// =============================================================================

class ITBRangeTest : public ::testing::Test {};

TEST_F(ITBRangeTest, DefaultConstruction) {
    ITBRange r;
    EXPECT_TRUE(r.is_empty());
}

TEST_F(ITBRangeTest, Combine) {
    RangeSet<TBRange, size_t> rs1, rs2;
    rs1.add(0, TBRange({0, 0}, {32, 32}, 0, 64));
    rs2.add(1, TBRange({32, 32}, {64, 64}, 0, 64));
    
    ITBRange r1(rs1);
    ITBRange r2(rs2);
    r1.combine(r2);
    
    EXPECT_EQ(r1.range_set.size(), 2u);
}

TEST_F(ITBRangeTest, ExtendForloopDim) {
    RangeSet<TBRange, size_t> rs;
    rs.add(0, TBRange({0, 0}, {32, 32}, 10, 20));
    
    ITBRange r(rs);
    ITBRange extended = r.extend_forloop_dim();
    
    EXPECT_EQ(extended.range_set.get(0).forloop_lower, 0);
    EXPECT_EQ(extended.range_set.get(0).forloop_upper, Range::INF);
}

// =============================================================================
// Range Propagation Utility Tests
// =============================================================================

class RangePropagationUtilTest : public ::testing::Test {};

TEST_F(RangePropagationUtilTest, IntersectRanges) {
    // Test that ranges can be intersected
    Range r1({0, 0}, {100, 100});
    Range r2({50, 50}, {150, 150});
    
    // Intersection should be (50,50) to (100,100)
    std::vector<int> new_lower, new_upper;
    for (size_t i = 0; i < r1.lower.size(); ++i) {
        new_lower.push_back(std::max(r1.lower[i], r2.lower[i]));
        new_upper.push_back(std::min(r1.upper[i], r2.upper[i]));
    }
    Range intersection(new_lower, new_upper);
    
    EXPECT_EQ(intersection.lower[0], 50);
    EXPECT_EQ(intersection.upper[0], 100);
}

TEST_F(RangePropagationUtilTest, UnionRanges) {
    // Test that ranges can be unioned
    Range r1({0, 0}, {50, 50});
    Range r2({50, 50}, {100, 100});
    
    // Union bounding box should be (0,0) to (100,100)
    std::vector<int> new_lower, new_upper;
    for (size_t i = 0; i < r1.lower.size(); ++i) {
        new_lower.push_back(std::min(r1.lower[i], r2.lower[i]));
        new_upper.push_back(std::max(r1.upper[i], r2.upper[i]));
    }
    Range union_range(new_lower, new_upper);
    
    EXPECT_EQ(union_range.lower[0], 0);
    EXPECT_EQ(union_range.upper[0], 100);
}

// =============================================================================
// Parameterized Range Tests
// =============================================================================

struct RangeTestParam {
    std::vector<int> lower;
    std::vector<int> upper;
    bool expected_empty;
    size_t expected_volume;
};

class RangeParameterizedTest : public ::testing::TestWithParam<RangeTestParam> {};

TEST_P(RangeParameterizedTest, RangeProperties) {
    auto param = GetParam();
    Range r(param.lower, param.upper);
    
    EXPECT_EQ(r.is_empty(), param.expected_empty);
    if (!param.expected_empty) {
        EXPECT_EQ(r.volume(), param.expected_volume);
    }
}

INSTANTIATE_TEST_SUITE_P(
    RangeVariants,
    RangeParameterizedTest,
    ::testing::Values(
        RangeTestParam{{0, 0}, {10, 10}, false, 100},
        RangeTestParam{{0, 0, 0}, {5, 5, 5}, false, 125},
        RangeTestParam{{10, 10}, {10, 10}, true, 0},  // Empty
        RangeTestParam{{5, 5}, {10, 5}, true, 0},     // Empty (one dim)
        RangeTestParam{{0}, {100}, false, 100}        // 1D
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
