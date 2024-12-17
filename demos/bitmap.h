#pragma once

#include <cassert>
#include <cstdint>
#include <new>

#include <roaring/roaring.h>

class RoaringBitmap {
   private:
    roaring_bitmap_t* bitmap;

   public:
    // 构造函数，初始化 bitmap，构造函数参数为容量
    explicit RoaringBitmap(uint32_t capacity = 0) {
        bitmap = roaring_bitmap_create_with_capacity(capacity);
        if (!bitmap) {
            throw std::bad_alloc();
        }
    }

    // 析构函数：释放内存
    ~RoaringBitmap() {
        roaring_bitmap_free(bitmap);
    }

    // 删除拷贝构造函数和拷贝赋值运算符
    RoaringBitmap(const RoaringBitmap&) = delete;
    RoaringBitmap& operator=(const RoaringBitmap&) = delete;

    // 删除移动构造函数和移动赋值运算符
    RoaringBitmap(RoaringBitmap&&) = delete;
    RoaringBitmap& operator=(RoaringBitmap&&) = delete;

    // 添加元素到 bitmap
    void add(uint32_t value) {
        roaring_bitmap_add(bitmap, value);
    }

    // 判断一个元素是否在 bitmap 中
    bool contains(uint32_t value) const {
        return roaring_bitmap_contains(bitmap, value);
    }

    // 删除 bitmap 中一个元素
    void remove(uint32_t value) {
        roaring_bitmap_remove(bitmap, value);
    }
};
