#pragma once

#include <iostream>
#include <glad/glad.h>
#include <fstream>
#include <fftw3.h>
#include <complex>
#include <nlohmann/json.hpp>
#include <filesystem>

using json = nlohmann::json;

inline fftwf_complex* fftwf_cast(const std::complex<float>* p) {
    return const_cast<fftwf_complex*>(reinterpret_cast<const fftwf_complex*>(p));
}

inline GLenum glCheckError_(const char* file, int line)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (errorCode)
        {
        case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
        case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
        case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
        case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
        case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
        case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
        case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        std::cout << error << " | " << file << " (" << line << ")" << std::endl;
    }
    return errorCode;
}
#define glCheckError() glCheckError_(__FILE__, __LINE__) 

template<class Matrix>
void WriteMatrix(std::string filename, const Matrix& matrix) {
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
    out.write((char*)(&rows), sizeof(typename Matrix::Index));
    out.write((char*)(&cols), sizeof(typename Matrix::Index));
    out.write((char*)matrix.data(), rows * cols * sizeof(typename Matrix::Scalar));
    out.close();
}

template<class Matrix>
void ReadMatrix(std::string filename, Matrix& matrix) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    typename Matrix::Index rows = 0, cols = 0;
    in.read((char*)(&rows), sizeof(typename Matrix::Index));
    in.read((char*)(&cols), sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read((char*)matrix.data(), rows * cols * sizeof(typename Matrix::Scalar));
    in.close();
}

inline bool ReadJson(std::string filename, json& j) {
    std::ifstream f(filename);
    if (f.peek() == std::ifstream::traits_type::eof()) return false;
    f >> j;
    f.close();
    return true;
}

inline void WriteJson(std::string filename, json& j) {
    std::ofstream f(filename);
    f << j.dump();
    f.close();
}