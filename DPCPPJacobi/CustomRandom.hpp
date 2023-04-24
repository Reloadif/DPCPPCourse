#include <random>
#include <exception>

const int MaxRandomValue = 100;

template <typename T>
void getRandomVector(T* vector, const int size) {
    if (vector == nullptr) throw std::exception("The pointer cannot be nullptr!");

    std::random_device dev;
    std::mt19937 gen(dev());

    for (int i = 0; i < size; ++i) {
        vector[i] = gen() % MaxRandomValue;
    }
}

template <typename T>
void getRandomMatrix(T* matrix, const int rows, const int columns) {
    if (matrix == nullptr) throw std::exception("The pointer cannot be nullptr!");

    std::random_device dev;
    std::mt19937 gen(dev());

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            T value = gen() % MaxRandomValue;
            while (value == 0) value = gen() % MaxRandomValue;

            matrix[i * columns + j] = value;
            if (i == j) {
                matrix[i * columns + j] += MaxRandomValue * columns * 2;
            }
        }
    }
}
