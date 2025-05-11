#ifndef DATATYPE_H
#define DATATYPE_H

#include <stdexcept>
#include <ATen/ATen.h>

// Define DataType enum
enum class DataType {
    FP32,
    FP16,
    BF16
};

// Map ATEN data types to DataType
inline DataType ATEN2DT(at::ScalarType DT) {
    switch (DT) {
        case at::kFloat:
            return DataType::FP32;
        case at::kHalf:
            return DataType::FP16;
        case at::kBFloat16:
            return DataType::BF16;
        default:
            throw std::invalid_argument("Unsupported at::ScalarType for DataType mapping.");
    }
}

inline const char* DT2STR(DataType DT) {
    switch (DT) {
        case DataType::FP32:
            return "FP32";
        case DataType::FP16:
            return "FP16";
        case DataType::BF16:
            return "BF16";
        default:
            return "Unknown";
    }
}


inline size_t SIZEOF(DataType DT) {
    switch (DT) {
        case DataType::FP32:
            return 4;
        case DataType::FP16:
            return 2;
        case DataType::BF16:
            return 2;
        default:
            throw std::runtime_error("Invalid DataType");
    }
}

#endif // DATATYPE_H